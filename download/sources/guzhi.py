# -*- coding: utf-8 -*-
"""东方财富估值走势抓取（接入模块，日频：市盈率/市净率/市销率/市现率）。

仅保留数据源特有逻辑：东财 RPT_CUSTOM_DMSK_TREND 接口请求（curl + 本地代理）、
4 个估值指标逐个拉取、(symbol, indicator) 双键水位过滤增量。
日志/建表/UPSERT/暂存路径/水位/合并/结果统一走 download.common。

用法（独立运行）：
    python -m download.sources.guzhi                          # 从 stock_list 抓取全部 A 股（近1年日频）
    python -m download.sources.guzhi --codes 300999,601318    # 指定股票代码
    python -m download.sources.guzhi --datetype 4             # 指定口径（4=近10年月频，仅手动回补用）
    python -m download.sources.guzhi --stage auto             # 暂存模式
    python -m download.sources.guzhi --merge "download/temp/guzhi_stage_*.duckdb"

接口（无需 Cookie / token，GET 即可）：
    估值走势 RPT_CUSTOM_DMSK_TREND: INDICATOR_VALUE 实际估值
    INDICATORTYPE: 1=市盈率 2=市净率 3=市销率 4=市现率（必填，不可省略）
    DATETYPE:      1=近1年(日频) 2=近3年(周频) 3=近5年(周频) 4=近10年(月频)
    注：原 gzfx 模块的估值通道接口（RPT_CUSTOM_DMSK，stock_price/pass1~5/mult1~5）已弃用，
    只保留 value 单列，与 TradingView 估值数据共用同一张 guzhi 表（UPSERT 幂等）。

入库表 guzhi 字段含义（与 download/tv/tradingview_guzhi.py 共用，主键 (symbol, indicator, date)）：
    symbol     股票代码（6 位数字）
    indicator  指标类型：pe=市盈率 pb=市净率 ps=市销率 pcf=市现率
    date       交易日期
    value      实际估值（PE/PB/PS/PCF 的 TTM 值，来自走势接口 INDICATOR_VALUE；空值不入库）

依赖：
    pip install duckdb
"""

import datetime
import json
import subprocess
import time

from download.common import cli
from download.common import logging as common_logging
from download.common import results
from download.common import shared
from download.common import storage
from download.common.paths import DEFAULT_DB_PATH, TEMP_DIR

SOURCE_NAME = "guzhi"
TABLE_NAME = "guzhi"
TABLE_SCHEMA = """(
    symbol TEXT,
    indicator TEXT,
    date DATE,
    value DOUBLE,
    PRIMARY KEY (symbol, indicator, date)
)"""
COLUMNS = ["symbol", "indicator", "date", "value"]

INDICATOR_TYPES = {1: "pe", 2: "pb", 3: "ps", 4: "pcf"}  # 市盈/市净/市销/市现
DATE_TYPES = {1: "1y", 2: "3y", 3: "5y", 4: "10y"}        # 1=日频 2/3=周频 4=月频
DEFAULT_DELAY = 0.3  # 每股抓取间隔（秒），避免触发东财限流

API_BASE = "https://datacenter.eastmoney.com/securities/api/data/get"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
PROXY = "http://127.0.0.1:7897"


def _to_date(s):
    """'2026-09-18 00:00:00' -> '2026-09-18'"""
    return (s or "").split(" ")[0]


def fetch_api(code, indicator_type, date_type, log):
    """GET 估值走势接口，返回 data 列表。失败返回 []。"""
    filt = f"(SECURITY_CODE%3D%22{code}%22)(INDICATORTYPE%3D{indicator_type})(DATETYPE%3D{date_type})"
    url = (
        f"{API_BASE}?type=RPT_CUSTOM_DMSK_TREND&p=1&sr=-1&st=TRADE_DATE"
        f"&var=source=DataCenter&client=WAP&filter={filt}"
    )
    try:
        r = subprocess.run(
            ["curl", "-s", "-f", "-x", PROXY, url,
             "-H", f"User-Agent: {UA}", "-H", "Accept: application/json"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            log.error("%s it=%s curl exit %s", code, indicator_type, r.returncode)
            return []
        j = json.loads(r.stdout)
        return (j.get("result") or {}).get("data") or []
    except Exception as e:
        log.error("%s it=%s 异常: %s", code, indicator_type, e)
        return []


def fetch_stock(code, date_type=1, min_dates=None, end=None, log=None):
    """抓单只股票 4 个估值指标的走势，返回 (symbol, indicator, date, value) 行列表。

    min_dates: {(symbol, indicator): 'YYYY-MM-DD'} 增量水位，date <= 水位的行跳过；
    end: 统一截止日 'YYYY-MM-DD'，超过 end 的新数据跳过不入库；
    value 为空的行不入库（亏损期 PE 等为空值）。
    """
    min_dates = min_dates or {}
    rows = []
    for it, iname in INDICATOR_TYPES.items():
        wm = min_dates.get((code, iname))
        for rec in fetch_api(code, it, date_type, log):
            d = _to_date(rec.get("TRADE_DATE"))
            v = rec.get("INDICATOR_VALUE")
            if not d or v is None:
                continue
            if end and d > end:
                continue
            if wm and d <= wm:
                continue
            rows.append((code, iname, d, v))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    return rows


merge_stages = storage.make_merge_stages(TABLE_NAME, TABLE_SCHEMA, COLUMNS, DEFAULT_DB_PATH)


def run(codes=None, date_type=1, end=None, db_path=None, delay=DEFAULT_DELAY,
        exclude_st=True, stage_path=None, run_id=""):
    """执行估值走势抓取，写入暂存库（或主库），不在此合并。

    参数：
        codes: 股票代码列表；为 None 时从 stock_list 表读在市股票（排除 ST）
        date_type: 抓取口径，默认 1=近1年(日频，每日增量)；2=近3年 3=近5年 4=近10年（手动回补用）
        end: 统一截止日期 YYYY-MM-DD；超过该日的新数据跳过不入库（None 则增量到最新）
        db_path: 主库路径，默认 download/autots.duckdb
        delay: 每股抓取间隔秒数（限速防封）
        exclude_st: stock_list 模式排除名称含 ST 的股票（默认 True）
        stage_path: 暂存 DuckDB 路径；"auto" 自动生成，None 直入主库；增量起点读主库+暂存库水位
    返回：
        RunResult（fetch_status/rows_staged/failed 等）
    """
    log = common_logging.get_logger(SOURCE_NAME, run_id)
    db_path = db_path or DEFAULT_DB_PATH
    res = results.RunResult(SOURCE_NAME, run_id)

    today = datetime.date.today()
    end = str(end).strip().replace("/", "-") if end else today.isoformat()
    end_date = datetime.date.fromisoformat(end)

    stage_path = storage.make_stage_path(SOURCE_NAME, TEMP_DIR, stage_path)
    if stage_path:
        log.info("暂存模式：写入 %s，事后用 merge 合并入主库", stage_path)
    res.stage_path = stage_path
    ingest_db = stage_path or db_path

    if isinstance(codes, str):
        codes = [c.strip() for c in codes.split(",") if c.strip()]

    stock_list_mode = codes is None
    if stock_list_mode:
        entries_all = shared.load_stock_list(db_path)
        n_st = sum(1 for e in entries_all if shared.is_st_name(e[2]))
        codes = [s for s, _d, n in entries_all if not exclude_st or not shared.is_st_name(n)]
        log.info("从 stock_list 读取 %d 只，排除 ST %d 只，待抓 %d 只（%s）",
                 len(entries_all), n_st, len(codes), DATE_TYPES[date_type])
    elif not codes:
        raise ValueError("未提供股票代码")

    # (symbol, indicator) 双键水位：主库 + 未合并暂存库合并取最大日期
    wm_rows = shared.load_max_dates(
        db_path, SOURCE_NAME, TABLE_NAME, ["symbol", "indicator"], stage_path=stage_path)
    min_dates = {(str(sym), ind): d.isoformat() for sym, ind, d in wm_rows}
    shared.log_max_date_distribution(log, min_dates, TABLE_NAME, unit="条序列")

    last_completed = shared.last_completed_trading_day(db_path, end_date)
    if last_completed is not None:
        log.info("最近已齐全交易日 %s，4 个指标均已覆盖该日的股票直接跳过（不发请求）",
                 last_completed.strftime("%Y-%m-%d"))

    total_rows = 0
    series = {}
    failed = {}
    no_new = 0
    skipped = 0

    for idx, code in enumerate(codes, 1):
        if last_completed is not None:
            covered = [min_dates.get((code, iname)) for iname in INDICATOR_TYPES.values()]
            if all(c and c >= last_completed.isoformat() for c in covered):
                skipped += 1
                continue
        try:
            rows = fetch_stock(code, date_type=date_type, min_dates=min_dates, end=end, log=log)
        except Exception as e:
            failed[code] = str(e)
            log.error("%s 失败: %s", code, e)
            time.sleep(delay)
            continue
        if not rows:
            # 库中已有该股数据且无新交易日 = 已最新，不算失败
            if any((code, iname) in min_dates for iname in INDICATOR_TYPES.values()):
                no_new += 1
                continue
            failed[code] = "无数据"
            log.warning("%s 无数据", code)
            time.sleep(delay)
            continue
        n = storage.ingest(ingest_db, TABLE_NAME, TABLE_SCHEMA, COLUMNS, rows)
        total_rows += n
        first, last = rows[0][2], rows[-1][2]
        series[code] = {"n": n, "first": first, "last": last}
        if idx % 100 == 0 or idx == len(codes):
            log.info("[%d/%d] 进度：成功 %d 失败 %d 无新数据 %d 累计入库 %d 行",
                     idx, len(codes), len(series), len(failed), no_new, total_rows)
        else:
            log.info("[%d/%d] %s 入库 %d 行 %s ~ %s", idx, len(codes), code, n, first, last)
        time.sleep(delay)

    if failed:
        res.failed_path = results.failed_path_for(SOURCE_NAME, TEMP_DIR)
        results.write_failed_list(failed, res.failed_path)
        log.warning("%d 只失败，清单: %s", len(failed), res.failed_path)

    log.info("完成：成功 %d 只，失败 %d 只，无新数据 %d 只，跳过(已最新) %d 只，累计入库 %d 行",
             len(series), len(failed), no_new, skipped, total_rows)

    res.rows_staged = total_rows
    res.success = len(series)
    res.failed = len(failed)
    res.no_new = no_new
    res.skipped = skipped
    res.detail = {"series": series}
    if failed:
        if not series and not no_new and not skipped:
            res.fetch_status = results.FETCH_FAILED
            res.error = f"全部失败: {failed}"
        else:
            res.fetch_status = results.FETCH_PARTIAL
    else:
        res.fetch_status = results.FETCH_OK
    return res.finish()


def _add_args(parser):
    parser.add_argument("--codes", default=None, help="逗号分隔股票代码，如 300999,601318；不传则从 stock_list 表读取全部")
    parser.add_argument("--datetype", type=int, default=1, choices=[1, 2, 3, 4],
                        help="抓取口径：1=近1年(日频,每日增量) 2=近3年(周频) 3=近5年(周频) 4=近10年(月频)，默认 1")
    parser.add_argument("--end", default=None, help="统一截止日期 YYYY-MM-DD（不传则增量到最新）")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help=f"每股间隔秒数，默认 {DEFAULT_DELAY}")
    parser.add_argument("--include-st", action="store_true", help="stock_list 模式下不排除 ST 股票")
    parser.add_argument("--stage", default=None, help="暂存 DuckDB 路径；auto 自动生成，不设则直入主库")


def _build_kwargs(args, db_path):
    return dict(
        codes=args.codes,
        date_type=args.datetype,
        end=args.end,
        db_path=db_path,
        delay=args.delay,
        exclude_st=not args.include_st,
        stage_path=args.stage,
    )


def _log_start(log, args, db_path):
    log.info("开始，主库: %s，口径: %s，排除 ST: %s，每股间隔 %ss",
             db_path, DATE_TYPES[args.datetype], not args.include_st, args.delay)


def main():
    cli.standard_main(
        source_name=SOURCE_NAME,
        description="东方财富估值走势抓取入库（PE/PB/PS/PCF，日频，入 guzhi 表）",
        add_args=_add_args,
        kwargs_builder=_build_kwargs,
        run_func=run,
        merge_func=merge_stages,
        default_db_path=DEFAULT_DB_PATH,
        temp_dir=TEMP_DIR,
        log_start=_log_start,
    )


if __name__ == "__main__":
    main()
