# -*- coding: utf-8 -*-
"""AKShare 腾讯证券 A 股历史行情抓取（接入模块）。

仅保留数据源特有逻辑：接口请求、复权方式、单位换算（volume 手→股）、
逐股增量起点（max(list_date, 库内最大日期+1)）。
日志/建表/UPSERT/暂存路径/水位/合并/结果统一走 download.common。

用法（独立运行）：
    python -m download.sources.akshare                          # stock_list 全量/增量
    python -m download.sources.akshare --codes 000001,600000    # 指定股票
    python -m download.sources.akshare --adjust qfq             # 前复权
    python -m download.sources.akshare --stage auto             # 暂存模式
    python -m download.sources.akshare --merge "download/temp/akshare_stage_*.duckdb"

入库表 akshare_tx 字段含义：
    symbol      股票代码（如 "000001"，自动补全市场前缀 sz/sh）
    date        交易日
    open        开盘价（元）
    close       收盘价（元）
    high        最高价（元）
    low         最低价（元）
    volume      成交量（股；腾讯接口原始单位为手，入库时已 ×100 换算）
    turnover    换手率（小数）
    amount      成交额（元）
    adjust      复权方式（""=不复权, qfq=前复权, hfq=后复权）
"""

import datetime
import time

from download.common import cli
from download.common import incremental
from download.common import logging as common_logging
from download.common import results
from download.common import shared
from download.common import storage
from download.common.errors import EmptyDataError
from download.common.paths import DEFAULT_DB_PATH, TEMP_DIR

try:
    import akshare as ak
except ImportError:
    ak = None

SOURCE_NAME = "akshare"
TABLE_NAME = "akshare_tx"
TABLE_SCHEMA = """(
    symbol TEXT,
    date DATE,
    open DOUBLE,
    close DOUBLE,
    high DOUBLE,
    low DOUBLE,
    volume DOUBLE,
    turnover DOUBLE,
    amount DOUBLE,
    adjust TEXT,
    PRIMARY KEY (symbol, date, adjust)
)"""
COLUMNS = ["symbol", "date", "open", "close", "high", "low", "volume", "turnover", "amount", "adjust"]

SHARES_PER_LOT = 100  # 腾讯接口 volume 单位为手，1 手 = 100 股
DEFAULT_DELAY = 1.0   # 每股抓取间隔（秒），避免触发腾讯限流
RETRY_TIMES = 2
RETRY_BACKOFF = 3.0


def _require_akshare():
    if ak is None:
        raise ImportError("缺少 akshare，请安装：python3 -m pip install akshare")


def fetch_hist(symbol, start_date, end_date, adjust="", timeout=30, retries=RETRY_TIMES):
    """调用 akshare 拉取历史行情，返回 DataFrame；空数据不重试，网络等异常按次数重试。"""
    _require_akshare()
    last_err = None
    for attempt in range(retries + 1):
        try:
            df = ak.stock_zh_a_hist_tx(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
                timeout=timeout,
            )
            if df is None or df.empty:
                raise EmptyDataError(f"{symbol}: 未返回数据")
            return df
        except EmptyDataError:
            raise
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(RETRY_BACKOFF * (attempt + 1))
    raise last_err


def df_to_rows(code, df, adjust=""):
    """DataFrame 转入库行 (symbol, date, open, close, high, low, volume, turnover, amount, adjust)。

    volume 原始单位为手，此处 ×100 统一为股。
    """
    rows = []
    for _, r in df.iterrows():
        rows.append((
            code,
            r["date"],
            float(r["open"]),
            float(r["close"]),
            float(r["high"]),
            float(r["low"]),
            float(r["volume"]) * SHARES_PER_LOT,
            float(r["turnover"]),
            float(r["amount"]),
            adjust,
        ))
    return rows


def load_stock_list(db_path=DEFAULT_DB_PATH, exclude_st=True):
    """从 stock_list 表读 (symbol, list_date, name)；list_date 缺失用 1990-01-01。

    exclude_st=True 时排除名称含 ST 的股票（含 *ST）。
    """
    entries = shared.load_stock_list(db_path)
    if exclude_st:
        entries = [e for e in entries if not shared.is_st_name(e[2])]
    return entries


merge_stages = storage.make_merge_stages(TABLE_NAME, TABLE_SCHEMA, COLUMNS, DEFAULT_DB_PATH)


def run(codes=None, start=None, end=None, adjust="hfq", db_path=None,
        codes_file=None, delay=DEFAULT_DELAY, exclude_st=True,
        stage_path=None, run_id=""):
    """执行 AKShare 行情抓取，写入暂存库（或主库），不在此合并。

    参数：
        codes: 股票代码列表；为 None 且未指定 codes_file 时，从 stock_list 表读在市股票
        start: 起始日期 YYYY-MM-DD；stock_list 模式为增量下限，指定股票时默认当年 1 月 1 日
        end: 结束日期 YYYY-MM-DD，默认今天
        adjust: 复权方式，""=不复权, qfq=前复权, hfq=后复权
        db_path: 主库路径，默认 download/autots.duckdb
        codes_file: 含 symbol 列的 CSV 文件路径
        delay: 每股间隔秒数（限速防封）
        exclude_st: stock_list 模式排除名称含 ST 的股票（默认 True）
        stage_path: 暂存 DuckDB 路径；"auto" 自动生成，None 直入主库；增量起点始终读主库
    返回：
        RunResult（fetch_status/rows_staged/failed 等）
    """
    log = common_logging.get_logger(SOURCE_NAME, run_id)
    db_path = db_path or DEFAULT_DB_PATH
    res = results.RunResult(SOURCE_NAME, run_id)

    stage_path = storage.make_stage_path(SOURCE_NAME, TEMP_DIR, stage_path)
    if stage_path:
        log.info("暂存模式：写入 %s，事后用 merge 合并入主库", stage_path)
    res.stage_path = stage_path
    ingest_db = stage_path or db_path

    if codes is None and codes_file:
        import pandas as pd
        df_codes = pd.read_csv(codes_file)
        if "symbol" not in df_codes.columns:
            raise ValueError(f"CSV 文件 {codes_file} 缺少 symbol 列")
        codes = df_codes["symbol"].astype(str).tolist()
        log.info("从 %s 读取 %d 只股票", codes_file, len(codes))
        if not codes:
            raise ValueError("CSV 文件中无股票代码")

    if isinstance(codes, str):
        codes = [c.strip() for c in codes.split(",") if c.strip()]

    today = datetime.date.today()
    end = (end or today.strftime("%Y%m%d")).replace("-", "")
    end_date = datetime.datetime.strptime(end, "%Y%m%d").date()

    stock_list_mode = codes is None
    last_completed = None
    lower_date = None
    if stock_list_mode:
        entries_all = load_stock_list(db_path, exclude_st=False)
        n_st = sum(1 for e in entries_all if shared.is_st_name(e[2]))
        entries = [(s, d) for s, d, n in entries_all if not exclude_st or not shared.is_st_name(n)]
        log.info("从 stock_list 读取 %d 只，排除 ST %d 只，待抓 %d 只", len(entries_all), n_st, len(entries))
        max_dates = shared.load_max_date_map(
            db_path, SOURCE_NAME, TABLE_NAME, "symbol", "date",
            where_clause="adjust = ?", where_params=[adjust], stage_path=stage_path,
            log=log, unit="只股票",
        )  # 增量起点读主库+暂存库水位，并打印最后日期分布
        last_completed = shared.last_completed_trading_day(db_path, end_date)
        if last_completed is not None:
            log.info("最近已齐全交易日 %s，已覆盖该日的股票直接跳过（不发请求）",
                     last_completed.strftime("%Y-%m-%d"))
        lower = start.replace("-", "") if start else None
        lower_date = datetime.datetime.strptime(lower, "%Y%m%d").date() if lower else None
    else:
        if not codes:
            raise ValueError("未提供股票代码")
        start = (start or f"{today.year}0101").replace("-", "")
        entries = [(c, None) for c in codes]
        max_dates = {}
        lower = None

    total_rows = 0
    series = {}
    failed = {}
    skipped = 0
    no_new = 0

    for idx, (code, list_date) in enumerate(entries, 1):
        symbol = shared.code_to_symbol(code)
        adjust_label = adjust or "不复权"

        if stock_list_mode:
            md = max_dates.get(code)
            if last_completed is not None and md is not None and md >= last_completed:
                skipped += 1
                continue
            sdate = incremental.incremental_start(list_date, md, lower_date)
            if sdate > end_date:
                skipped += 1
                continue
            start_i = sdate.strftime("%Y%m%d")
        else:
            start_i = start

        try:
            df = fetch_hist(symbol, start_i, end, adjust=adjust)
        except EmptyDataError as e:
            # 库中已有该股数据时空返回 = 没有新交易日，不算失败
            if stock_list_mode and code in max_dates:
                no_new += 1
                continue
            failed[code] = str(e)
            log.warning("%s 无数据: %s", code, e)
            time.sleep(delay)
            continue
        except Exception as e:
            failed[code] = str(e)
            log.error("%s 失败: %s", code, e)
            time.sleep(delay)
            continue
        rows = df_to_rows(code, df, adjust=adjust)
        if not rows:
            failed[code] = "无数据"
            log.warning("%s 无数据", code)
            time.sleep(delay)
            continue
        n = storage.ingest(ingest_db, TABLE_NAME, TABLE_SCHEMA, COLUMNS, rows)
        total_rows += n
        closes = [r[3] for r in rows]
        series[code] = {
            "n": len(rows),
            "first": rows[0][1],
            "last": rows[-1][1],
            "mean_close": sum(closes) / len(closes) if closes else 0,
        }
        if idx % 100 == 0 or idx == len(entries):
            log.info("[%d/%d] 进度：成功 %d 失败 %d 无新数据 %d 累计入库 %d 行",
                     idx, len(entries), len(series), len(failed), no_new, total_rows)
        else:
            log.info("[%d/%d] %s (%s) %s 入库 %d 行  %s ~ %s",
                     idx, len(entries), code, symbol, adjust_label, n, rows[0][1], rows[-1][1])
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
    parser.add_argument("--codes", default=None, help="逗号分隔股票代码，如 000001,600000；不传则从 stock_list 表读取全部")
    parser.add_argument("--codes-file", default=None, help="CSV 文件路径，包含 symbol 列的股票代码列表")
    parser.add_argument("--start", default=None, help="起始日期 YYYY-MM-DD（stock_list 模式作下限）")
    parser.add_argument("--end", default=None, help="结束日期 YYYY-MM-DD（默认今天）")
    parser.add_argument("--adjust", default="hfq", choices=["", "qfq", "hfq"], help="复权方式（默认 hfq）")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help=f"每股间隔秒数，默认 {DEFAULT_DELAY}")
    parser.add_argument("--include-st", action="store_true", help="stock_list 模式下不排除 ST 股票")
    parser.add_argument("--stage", default=None, help="暂存 DuckDB 路径；auto 自动生成，不设则直入主库")


def _build_kwargs(args, db_path):
    return dict(
        codes=args.codes,
        start=args.start,
        end=args.end,
        adjust=args.adjust,
        db_path=db_path,
        codes_file=args.codes_file,
        delay=args.delay,
        exclude_st=not args.include_st,
        stage_path=args.stage,
    )


def _log_start(log, args, db_path):
    log.info("开始，主库: %s，复权: %s，排除 ST: %s，每股间隔 %ss",
             db_path, args.adjust or "不复权", not args.include_st, args.delay)


def main():
    cli.standard_main(
        source_name=SOURCE_NAME,
        description="AKShare 腾讯证券 A 股历史行情抓取入库",
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
