# -*- coding: utf-8 -*-
"""东方财富个股融资融券抓取（接入模块，日频）。

仅保留数据源特有逻辑：东财 RPT_MARGIN_STATISTICS_STOCKS 接口请求（curl + 本地代理）、
按 TRADE_DATE 倒序翻页、min_date 截止增量（翻到 <= 截止日的旧数据即停）、
无两融数据股票黑名单（确认无数据自动写入，后续运行直接跳过，不再发请求）。
日志/建表/UPSERT/暂存路径/水位/合并/结果统一走 download.common。

用法（独立运行）：
    python -m download.sources.rzrq                          # 从 stock_list 抓取全部 A 股两融明细（自动跳过黑名单）
    python -m download.sources.rzrq --codes 688223,300999    # 指定股票代码
    python -m download.sources.rzrq --stage auto             # 暂存模式
    python -m download.sources.rzrq --merge "download/temp/rzrq_stage_*.duckdb"

接口（无需 Cookie / token，GET 即可）：
    RPT_MARGIN_STATISTICS_STOCKS，按 TRADE_DATE 倒序分页，ps 最大 500
    filter=(SECURITY_CODE="688223")

入库表 rzrq 字段含义（金额单位：元；量单位：股；比率为 %）：
    code                 股票代码（6 位数字）
    name                 股票简称
    date                 交易日期
    margin_balance       两融余额（融资余额+融券余额，元）
    margin_balance_ratio 两融余额占流通市值比（%）
    fin_balance          融资余额（元）
    fin_balance_ratio    融资余额占流通市值比（%）
    loan_balance         融券余额（元）
    loan_balance_ratio   融券余额占流通市值比（%）
    fin_netbuy_amt       融资净买入额（融资买入-融资偿还，元）
    fin_tval_ratio       融资净买入额占成交额比（%）
    fin_buy_amt          融资买入额（元）
    fin_repay_amt        融资偿还额（元）
    loan_balance_vol     融券余量（股）
    loan_netsell_amt     融券净卖出额（元）
    loan_tval_ratio      融券净卖出额占成交额比（%）
    loan_netsell_vol     融券净卖出量（股）
    loan_sell_vol        融券卖出量（股）
    loan_repay_vol       融券偿还量（股）
"""

import json
import os
import subprocess
import time

from download.common import blacklist as blacklist_common
from download.common import cli
from download.common import logging as common_logging
from download.common import results
from download.common import shared
from download.common import storage
from download.common.paths import DEFAULT_DB_PATH, DOWNLOAD_DIR, TEMP_DIR

SOURCE_NAME = "rzrq"
TABLE_NAME = "rzrq"
TABLE_SCHEMA = """(
    code TEXT,
    name TEXT,
    date DATE,
    margin_balance DOUBLE,
    margin_balance_ratio DOUBLE,
    fin_balance DOUBLE,
    fin_balance_ratio DOUBLE,
    loan_balance DOUBLE,
    loan_balance_ratio DOUBLE,
    fin_netbuy_amt DOUBLE,
    fin_tval_ratio DOUBLE,
    fin_buy_amt DOUBLE,
    fin_repay_amt DOUBLE,
    loan_balance_vol DOUBLE,
    loan_netsell_amt DOUBLE,
    loan_tval_ratio DOUBLE,
    loan_netsell_vol DOUBLE,
    loan_sell_vol DOUBLE,
    loan_repay_vol DOUBLE,
    PRIMARY KEY (code, date)
)"""
COLUMNS = [
    "code", "name", "date",
    "margin_balance", "margin_balance_ratio",
    "fin_balance", "fin_balance_ratio", "loan_balance", "loan_balance_ratio",
    "fin_netbuy_amt", "fin_tval_ratio", "fin_buy_amt", "fin_repay_amt",
    "loan_balance_vol", "loan_netsell_amt", "loan_tval_ratio",
    "loan_netsell_vol", "loan_sell_vol", "loan_repay_vol",
]

PAGE_SIZE = 500
DEFAULT_DELAY = 0.3  # 每股抓取间隔（秒），避免触发东财限流

API_BASE = "https://datacenter.eastmoney.com/securities/api/data/get"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
PROXY = "http://127.0.0.1:7897"

DEFAULT_BLACKLIST_FILE = os.path.join(DOWNLOAD_DIR, "rzrq_blacklist.txt")

# 接口请求字段（sty 参数），前 4 个为表头字段，其余与入库字段一一对应
API_COLUMNS = [
    "SECUCODE", "SECURITY_CODE", "TRADE_DATE", "SECURITY_NAME_ABBR",
    "MARGIN_BALANCE", "MARGIN_BALANCE_RATIO",
    "FIN_BALANCE", "FIN_BALANCE_RATIO", "LOAN_BALANCE", "LOAN_BALANCE_RATIO",
    "FIN_NETBUY_AMT", "FIN_TVAL_RATIO", "FIN_BUY_AMT", "FIN_REPAY_AMT",
    "LOAN_BALANCE_VOL", "LOAN_NETSELL_AMT", "LOAN_TVAL_RATIO",
    "LOAN_NETSELL_VOL", "LOAN_SELL_VOL", "LOAN_REPAY_VOL",
]
# 与入库字段（COLUMNS 去掉 code/name/date 表头）顺序对应的接口字段
FIELD_KEYS = [
    "MARGIN_BALANCE", "MARGIN_BALANCE_RATIO",
    "FIN_BALANCE", "FIN_BALANCE_RATIO", "LOAN_BALANCE", "LOAN_BALANCE_RATIO",
    "FIN_NETBUY_AMT", "FIN_TVAL_RATIO", "FIN_BUY_AMT", "FIN_REPAY_AMT",
    "LOAN_BALANCE_VOL", "LOAN_NETSELL_AMT", "LOAN_TVAL_RATIO",
    "LOAN_NETSELL_VOL", "LOAN_SELL_VOL", "LOAN_REPAY_VOL",
]


def _to_date(s):
    """'2026-09-18 00:00:00' -> '2026-09-18'"""
    return (s or "").split(" ")[0]


def load_blacklist(path=DEFAULT_BLACKLIST_FILE):
    """从黑名单文件读取无两融数据股票代码集合（一行一个代码，# 开头为注释；代码后可附名称，匹配只取首列）。文件不存在返回空集。"""
    return blacklist_common.load(path)


def add_to_blacklist(entries, path=DEFAULT_BLACKLIST_FILE):
    """把确认无两融数据的股票追加到黑名单文件（按代码去重，已在文件中的跳过），返回本次新增写入的数量。

    entries 为 (code, name) 元组，写入格式 "代码 名称"（名称为空则只写代码），读取时只取首列代码。
    """
    return blacklist_common.add(entries, path)


def fetch_page(code, page, log):
    """抓单页。成功返回 (data列表, 总页数)（data 为空表示该股无两融数据）；请求失败返回 None。"""
    url = (
        f"{API_BASE}?type=RPT_MARGIN_STATISTICS_STOCKS&sty={','.join(API_COLUMNS)}"
        f"&p={page}&ps={PAGE_SIZE}&sr=-1&st=TRADE_DATE&source=DataCenter&client=WAP"
        f"&filter=(SECURITY_CODE=%22{code}%22)"
    )
    try:
        r = subprocess.run(
            ["curl", "-s", "-f", "-x", PROXY, url,
             "-H", f"User-Agent: {UA}", "-H", "Accept: application/json"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            log.error("%s p%s curl exit %s", code, page, r.returncode)
            return None
        j = json.loads(r.stdout)
        res = j.get("result") or {}
        return res.get("data") or [], res.get("pages") or 0
    except Exception as e:
        log.error("%s p%s 异常: %s", code, page, e)
        return None


def fetch_stock(code, min_date=None, end=None, log=None):
    """抓单只股票两融数据（自动翻页）。

    min_date 给定时（如 "2026-09-22"），按 TRADE_DATE 倒序翻到 <= min_date 的旧数据即停，
    只返回更新日期，用于每日增量。
    end 给定时（统一截止日 "YYYY-MM-DD"），超过 end 的新数据直接跳过不入库。
    返回行列表；首页请求失败返回 None（接口/网络异常，调用方记失败，不入黑名单），
    接口正常但无数据返回 []（确认无两融数据，调用方入黑名单）。
    """
    rows = []
    page = 1
    pages = 1
    while page <= pages:
        ret = fetch_page(code, page, log)
        if ret is None:
            if page == 1:
                return None
            break  # 后续页失败保留已抓数据，不误判为无数据
        data, pages = ret
        if not data:
            break
        for rec in data:
            d = _to_date(rec.get("TRADE_DATE"))
            if end and d and d > end:
                continue
            if min_date and d and d <= min_date:
                rows.sort(key=lambda r: (r[0], r[2]))
                return rows
            rows.append(tuple(
                [code, rec.get("SECURITY_NAME_ABBR"), d]
                + [rec.get(k) for k in FIELD_KEYS]
            ))
        page += 1
    rows.sort(key=lambda r: (r[0], r[2]))
    return rows


def fetch_latest_date(log):
    """探测接口全局最新交易日（ps=1 取第一条 TRADE_DATE），用于跳过已最新股票。

    返回 'YYYY-MM-DD' 字符串；探测失败返回 None（调用方回退到逐只请求）。
    """
    url = (
        f"{API_BASE}?type=RPT_MARGIN_STATISTICS_STOCKS&sty=TRADE_DATE"
        f"&p=1&ps=1&sr=-1&st=TRADE_DATE&source=DataCenter&client=WAP"
    )
    try:
        r = subprocess.run(
            ["curl", "-s", "-f", "-x", PROXY, url,
             "-H", f"User-Agent: {UA}", "-H", "Accept: application/json"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            log.error("探测最新交易日 curl exit %s", r.returncode)
            return None
        j = json.loads(r.stdout)
        data = (j.get("result") or {}).get("data") or []
        if not data:
            return None
        return _to_date(data[0].get("TRADE_DATE")) or None
    except Exception as e:
        log.error("探测最新交易日异常: %s", e)
        return None


merge_stages = storage.make_merge_stages(TABLE_NAME, TABLE_SCHEMA, COLUMNS, DEFAULT_DB_PATH)


def run(codes=None, end=None, db_path=None, delay=DEFAULT_DELAY, exclude_st=True,
        blacklist_file=DEFAULT_BLACKLIST_FILE, stage_path=None, run_id=""):
    """执行融资融券抓取，写入暂存库（或主库），不在此合并。

    参数：
        codes: 股票代码列表；为 None 时从 stock_list 表读在市股票（排除 ST）
        end: 统一截止日期 YYYY-MM-DD；超过该日的新数据跳过不入库（None 则增量到最新）
        db_path: 主库路径，默认 download/autots.duckdb
        delay: 每股抓取间隔秒数（限速防封）
        exclude_st: stock_list 模式排除名称含 ST 的股票（默认 True）
        blacklist_file: 无两融数据股票黑名单文件（一行一个代码，可附名称）；黑名单内股票直接跳过，
            接口确认无数据的股票自动写入；None/"None" 则禁用
        stage_path: 暂存 DuckDB 路径；"auto" 自动生成，None 直入主库；增量起点读主库+暂存库水位
    返回：
        RunResult（fetch_status/rows_staged/failed 等）
    """
    log = common_logging.get_logger(SOURCE_NAME, run_id)
    db_path = db_path or DEFAULT_DB_PATH
    res = results.RunResult(SOURCE_NAME, run_id)

    if end:
        end = str(end).strip().replace("/", "-")
    if blacklist_file in ("", "None", "none"):
        blacklist_file = None

    stage_path = storage.make_stage_path(SOURCE_NAME, TEMP_DIR, stage_path)
    if stage_path:
        log.info("暂存模式：写入 %s，事后用 merge 合并入主库", stage_path)
    res.stage_path = stage_path
    ingest_db = stage_path or db_path

    if isinstance(codes, str):
        codes = [c.strip() for c in codes.split(",") if c.strip()]

    stock_list_mode = codes is None
    name_map = {}
    if stock_list_mode:
        entries_all = shared.load_stock_list(db_path)
        n_st = sum(1 for e in entries_all if shared.is_st_name(e[2]))
        codes = [s for s, _d, n in entries_all if not exclude_st or not shared.is_st_name(n)]
        name_map = {s: n for s, _d, n in entries_all}
        log.info("从 stock_list 读取 %d 只，排除 ST %d 只，待抓 %d 只", len(entries_all), n_st, len(codes))
    elif not codes:
        raise ValueError("未提供股票代码")

    blacklist = load_blacklist(blacklist_file)
    n_blacklisted = 0
    if blacklist:
        before = len(codes)
        codes = [c for c in codes if c not in blacklist]
        n_blacklisted = before - len(codes)
        if n_blacklisted:
            log.info("黑名单排除 %d 只无两融数据股票，剩余 %d 只", n_blacklisted, len(codes))

    max_dates = shared.load_max_date_map(
        db_path, SOURCE_NAME, TABLE_NAME, "code", as_string=True, stage_path=stage_path,
        log=log, unit="只股票",
    )

    latest_api = fetch_latest_date(log)
    _cands = [x for x in (latest_api, end) if x]
    effective_latest = min(_cands) if _cands else None
    if latest_api:
        log.info("接口最新交易日 %s，库内已覆盖到 %s 的股票直接跳过（不发请求）",
                 latest_api, effective_latest)

    total_rows = 0
    series = {}
    failed = {}
    no_new = 0
    skipped = 0
    no_data_new = []  # 本次接口确认无两融数据的 (code, name)（已即时写入黑名单，此处仅用于统计）

    for idx, code in enumerate(codes, 1):
        # 无新数据/跳过分支自身无日志，统一在循环入口每 100 只打点，避免长时间静默看似卡死
        if idx % 100 == 0 or idx == len(codes):
            log.info("[%d/%d] 进度：成功 %d 失败 %d 无新数据 %d 跳过(已最新) %d 累计入库 %d 行",
                     idx, len(codes), len(series), len(failed), no_new, skipped, total_rows)
        min_date = max_dates.get(code)
        if effective_latest and min_date and min_date >= effective_latest:
            skipped += 1
            continue
        try:
            rows = fetch_stock(code, min_date=min_date, end=end, log=log)
        except Exception as e:
            failed[code] = str(e)
            log.error("%s 失败: %s", code, e)
            time.sleep(delay)
            continue
        if rows is None:
            failed[code] = "请求失败"
            time.sleep(delay)
            continue
        if not rows:
            # 库中已有该股数据且无新交易日 = 已最新，不算失败
            if code in max_dates:
                no_new += 1
                continue
            # 接口正常返回但无任何两融数据 = 非两融标的，即时写入黑名单防止中断丢失
            entry = (code, name_map.get(code, ""))
            add_to_blacklist([entry], blacklist_file)
            no_data_new.append(entry)
            log.warning("%s 无两融数据，加入黑名单", code)
            time.sleep(delay)
            continue
        n = storage.ingest(ingest_db, TABLE_NAME, TABLE_SCHEMA, COLUMNS, rows)
        total_rows += n
        series[code] = {"n": n, "first": rows[0][2], "last": rows[-1][2]}
        log.info("[%d/%d] %s 入库 %d 行 %s ~ %s", idx, len(codes), code, n, rows[0][2], rows[-1][2])
        time.sleep(delay)

    if no_data_new:
        log.warning("本次 %d 只确认无两融数据，已即时写入黑名单 %s",
                    len(no_data_new), blacklist_file)

    if failed:
        res.failed_path = results.failed_path_for(SOURCE_NAME, TEMP_DIR)
        results.write_failed_list(failed, res.failed_path)
        log.warning("%d 只失败，清单: %s", len(failed), res.failed_path)

    log.info("完成：成功 %d 只，失败 %d 只，无新数据 %d 只，跳过(已最新) %d 只，"
             "黑名单排除 %d 只，新入黑名单 %d 只，累计入库 %d 行",
             len(series), len(failed), no_new, skipped, n_blacklisted, len(no_data_new), total_rows)

    res.rows_staged = total_rows
    res.success = len(series)
    res.failed = len(failed)
    res.no_new = no_new
    res.skipped = skipped
    res.detail = {"series": series, "blacklisted": n_blacklisted, "no_data": len(no_data_new)}
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
    parser.add_argument("--codes", default=None, help="逗号分隔股票代码，如 688223,300999；不传则从 stock_list 表读取全部")
    parser.add_argument("--end", default=None, help="统一截止日期 YYYY-MM-DD（不传则增量到最新）")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help=f"每股间隔秒数，默认 {DEFAULT_DELAY}")
    parser.add_argument("--include-st", action="store_true", help="stock_list 模式下不排除 ST 股票")
    parser.add_argument("--blacklist-file", default=DEFAULT_BLACKLIST_FILE,
                        help=f"无两融数据股票黑名单文件路径（一行一个代码，可附名称），默认 {DEFAULT_BLACKLIST_FILE}；None 则禁用")
    parser.add_argument("--stage", default=None, help="暂存 DuckDB 路径；auto 自动生成，不设则直入主库")


def _build_kwargs(args, db_path):
    return dict(
        codes=args.codes,
        end=args.end,
        db_path=db_path,
        delay=args.delay,
        exclude_st=not args.include_st,
        blacklist_file=args.blacklist_file,
        stage_path=args.stage,
    )


def _log_start(log, args, db_path):
    log.info("开始，主库: %s，排除 ST: %s，每股间隔 %ss",
             db_path, not args.include_st, args.delay)


def main():
    cli.standard_main(
        source_name=SOURCE_NAME,
        description="东方财富个股融资融券抓取入库（日频）",
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
