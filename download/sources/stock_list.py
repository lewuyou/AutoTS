# -*- coding: utf-8 -*-
"""A 股股票列表抓取（接入模块，快照表：手动定期触发，每次全量重建）。

仅保留数据源特有逻辑：akshare 沪深列表接口请求、腾讯全市场行情补齐市值/最新价、
简称清洗（XD/N/C/S 前缀）、股本推算（市值/最新价）。
日志/建表/暂存路径/整表重建与替换合并/结果统一走 download.common。

快照表：不参与 daily.py 默认批量任务，仅手动触发（python daily.py --sources stock_list）；
每次运行全量重建（整表替换合并，自动剔除退市股）。
其他数据源（akshare/baidu/rzrq/nbjb/guzhi）依赖本表筛选在市股票。

用法（独立运行）：
    python -m download.sources.stock_list                       # 抓取全部 A 股，整表重建入主库
    python -m download.sources.stock_list --stage auto          # 暂存模式
    python -m download.sources.stock_list --merge "download/temp/stock_list_stage_*.duckdb"

入库表 stock_list 字段含义：
    symbol          股票代码（如 "000001"）
    name            股票名称/简称（如 "平安银行"；已去除空格及 C/N/S/XD 特殊前缀，ST/*ST 保留）
    exchange        交易所（SH=上交所, SZ=深交所）
    full_name       证券全称（仅 SH）
    company_short   公司简称（仅 SH）
    company_full    公司全称（仅 SH）
    list_date       上市日期
    board           板块（SH: 主板/科创板, SZ: 主板/创业板）
    total_share     A股总股本（单位：股，整数，SZ 来自接口，SH 由市值/最新价推算）
    float_share     A股流通股本（单位：股，整数，SZ 来自接口，SH 由市值/最新价推算）
    industry        所属行业（仅 SZ）
    total_mv        总市值（单位：元，整数，来自 stock_zh_a_spot_tx，亿元换算后四舍五入取整）
    float_mv        流通市值（单位：元，整数，来自 stock_zh_a_spot_tx，亿元换算后四舍五入取整）
    last_price      最新价（单位：元，来自 stock_zh_a_spot_tx）

依赖：
    pip install akshare duckdb
"""

from collections import Counter

from download.common import cli
from download.common import logging as common_logging
from download.common import results
from download.common import storage
from download.common.paths import DEFAULT_DB_PATH, TEMP_DIR

try:
    import akshare as ak
except ImportError:
    ak = None

SOURCE_NAME = "stock_list"
TABLE_NAME = "stock_list"
TABLE_SCHEMA = """(
    symbol TEXT,
    name TEXT,
    exchange TEXT,
    full_name TEXT,
    company_short TEXT,
    company_full TEXT,
    list_date DATE,
    board TEXT,
    total_share BIGINT,
    float_share BIGINT,
    industry TEXT,
    total_mv BIGINT,
    float_mv BIGINT,
    last_price DOUBLE,
    PRIMARY KEY (symbol)
)"""
COLUMNS = [
    "symbol", "name", "exchange", "full_name", "company_short", "company_full",
    "list_date", "board", "total_share", "float_share", "industry",
    "total_mv", "float_mv", "last_price",
]

YI_TO_YUAN = 100000000  # 腾讯接口市值单位为亿元，入库统一为元


def _require_akshare():
    if ak is None:
        raise ImportError("缺少 akshare，请安装：python3 -m pip install akshare")


def _s(value):
    """转字符串，NaN/空值返回 None（避免 pandas NaN 被转成字符串 "nan" 入库）。"""
    if value is None or value != value:
        return None
    return str(value)


def _parse_share(value):
    """解析股本字符串（如 "19,405,918,198"）为整数，失败返回 None。"""
    if value is None or value != value:
        return None
    try:
        return int(str(value).replace(",", ""))
    except (ValueError, AttributeError):
        return None


def _to_list_date(value):
    """上市日期统一为 'YYYY-MM-DD' 字符串（date 对象/YYYYMMDD 字符串均可），空返回 None。"""
    s = _s(value)
    if s is None:
        return None
    s = s.strip().replace("/", "-")[:10]
    if len(s) == 8 and s.isdigit():
        s = f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return s


def clean_stock_name(name, company_short=None):
    """清理股票简称：去空格，去除特殊前缀标识，返回可用于搜索的完整简称。

    上交所证券简称有 4 字上限，除权(XD)/上市首日(N)/新股(C)/未股改(S)等临时前缀
    会挤占长度使主体被截断（如 "XD三元股" → 主体只剩 "三元股"）。此时改用公司简称
    字段取完整名（如 "三元股份"）。
    ST/*ST 为退市风险标识，按约定保留不处理；TCL 等真实字母缩写不是标识，保留。
    """
    def _strip(raw):
        if raw is None:
            return None
        raw = str(raw).replace(" ", "").strip()
        for prefix in ("XD", "N", "C"):
            if raw.startswith(prefix):
                raw = raw[len(prefix):]
                break
        # S 仅指未股改标识（后接中文简称），需排除 ST/*ST
        if raw.startswith("S") and not raw.startswith("ST"):
            raw = raw[1:]
        return raw

    if name is None:
        return None
    raw = str(name).replace(" ", "").strip()
    has_prefix = raw.startswith(("XD", "N", "C")) or (raw.startswith("S") and not raw.startswith("ST"))
    if has_prefix and company_short is not None:
        base = _strip(company_short)
        if base:
            return base
    return _strip(raw)


def fetch_sh_stocks(symbol, board, log):
    """上交所指定板块（"主板A股"/"科创板"）股票列表，返回 dict 列表。"""
    df = ak.stock_info_sh_name_code(symbol=symbol)
    stocks = []
    for _, row in df.iterrows():
        stocks.append({
            "symbol": str(row["证券代码"]).zfill(6),
            "name": clean_stock_name(row["证券简称"], company_short=row["公司简称"]),
            "exchange": "SH",
            "full_name": _s(row["证券全称"]),
            "company_short": _s(row["公司简称"]),
            "company_full": _s(row["公司全称"]),
            "list_date": _to_list_date(row["上市日期"]),
            "board": board,
            "total_share": None,
            "float_share": None,
            "industry": None,
        })
    log.info("上交所%s %d 只", board, len(stocks))
    return stocks


def fetch_sz_stocks(log):
    """深交所股票列表，返回 dict 列表（股本/行业来自接口）。"""
    df = ak.stock_info_sz_name_code()
    stocks = []
    for _, row in df.iterrows():
        stocks.append({
            "symbol": str(row["A股代码"]).zfill(6),
            "name": clean_stock_name(row["A股简称"]),
            "exchange": "SZ",
            "full_name": None,
            "company_short": None,
            "company_full": None,
            "list_date": _to_list_date(row["A股上市日期"]),
            "board": _s(row["板块"]),
            "total_share": _parse_share(row["A股总股本"]),
            "float_share": _parse_share(row["A股流通股本"]),
            "industry": _s(row["所属行业"]),
        })
    log.info("深交所 %d 只", len(stocks))
    return stocks


def fetch_spot_map(log):
    """腾讯全市场行情 {code: {total_mv, float_mv, last_price}}，市值单位已从亿元换算为元并取整。"""
    log.info("获取全市场市值数据（stock_zh_a_spot_tx） ...")
    df = ak.stock_zh_a_spot_tx()
    spot_map = {}
    for _, row in df.iterrows():
        # 去掉 sh/sz 前缀，统一为 6 位代码
        code = str(row["code"]).replace("sh", "").replace("sz", "").zfill(6)
        try:
            last_price = float(row["zxj"]) if row["zxj"] else None
        except (ValueError, TypeError):
            last_price = None
        try:
            total_mv = int(round(float(row["zsz"]) * YI_TO_YUAN)) if row["zsz"] else None
        except (ValueError, TypeError):
            total_mv = None
        try:
            float_mv = int(round(float(row["ltsz"]) * YI_TO_YUAN)) if row["ltsz"] else None
        except (ValueError, TypeError):
            float_mv = None
        spot_map[code] = {"total_mv": total_mv, "float_mv": float_mv, "last_price": last_price}
    log.info("全市场行情 %d 条", len(spot_map))
    return spot_map


def fetch_stock_list(log):
    """拉取沪深 A 股股票列表，返回 dict 列表（COLUMNS 除市值外字段 + total_mv/float_mv/last_price）。

    市值和最新价用 stock_zh_a_spot_tx 一次性获取并合并；
    SH 股本由市值/最新价推算，SZ 优先用接口数据、缺失时推算。
    """
    _require_akshare()

    stocks = fetch_sh_stocks("主板A股", "主板", log)
    stocks += fetch_sh_stocks("科创板", "科创板", log)
    stocks += fetch_sz_stocks(log)

    spot_map = fetch_spot_map(log)
    for s in stocks:
        spot = spot_map.get(s["symbol"])
        if spot:
            s.update(spot)
        else:
            s.update({"total_mv": None, "float_mv": None, "last_price": None})

    # 股本推算：市值 / 最新价（SH 全部推算，SZ 接口数据缺失时推算），四舍五入取整
    for s in stocks:
        if s["last_price"] and s["last_price"] > 0:
            if s["total_mv"] and not s["total_share"]:
                s["total_share"] = int(round(s["total_mv"] / s["last_price"]))
            if s["float_mv"] and not s["float_share"]:
                s["float_share"] = int(round(s["float_mv"] / s["last_price"]))

    return stocks


def stocks_to_rows(stocks):
    """dict 列表转 COLUMNS 顺序的入库行。"""
    return [tuple(s[c] for c in COLUMNS) for s in stocks]


merge_stages = storage.make_replace_merge_stages(TABLE_NAME, TABLE_SCHEMA, COLUMNS, DEFAULT_DB_PATH)


def run(db_path=None, stage_path=None, run_id=""):
    """执行股票列表抓取：直入主库时整表重建，暂存模式写暂存库（事后整表替换合并）。

    参数：
        db_path: 主库路径，默认 download/autots.duckdb
        stage_path: 暂存 DuckDB 路径；"auto" 自动生成，None 直入主库（整表重建）
    返回：
        RunResult（fetch_status/rows_staged 等）
    """
    log = common_logging.get_logger(SOURCE_NAME, run_id)
    db_path = db_path or DEFAULT_DB_PATH
    res = results.RunResult(SOURCE_NAME, run_id)

    stage_path = storage.make_stage_path(SOURCE_NAME, TEMP_DIR, stage_path)
    if stage_path:
        log.info("暂存模式：写入 %s，事后用 merge 整表替换入主库", stage_path)
    res.stage_path = stage_path

    log.info("抓取全部 A 股股票列表 ...")
    stocks = fetch_stock_list(log)
    if not stocks:
        raise RuntimeError("无数据")
    rows = stocks_to_rows(stocks)

    if stage_path:
        n = storage.ingest(stage_path, TABLE_NAME, TABLE_SCHEMA, COLUMNS, rows)
    else:
        n = storage.replace_table(db_path, TABLE_NAME, TABLE_SCHEMA, COLUMNS, rows)
    res.rows_staged = n
    res.success = n

    exchanges = Counter(s["exchange"] for s in stocks)
    log.info("交易所分布：%s", "，".join(f"{ex} {c} 只" for ex, c in sorted(exchanges.items())))
    log.info("完成：共 %d 只，写入 %d 行", len(stocks), n)

    res.detail = {"exchanges": dict(exchanges)}
    res.fetch_status = results.FETCH_OK
    return res.finish()


def _add_args(parser):
    parser.add_argument("--stage", default=None, help="暂存 DuckDB 路径；auto 自动生成，不设则直入主库（整表重建）")


def _build_kwargs(args, db_path):
    return dict(db_path=db_path, stage_path=args.stage)


def _log_start(log, args, db_path):
    log.info("开始，主库: %s，快照表全量重建", db_path)


def main():
    cli.standard_main(
        source_name=SOURCE_NAME,
        description="A 股股票列表抓取入库（快照表，每次全量重建，入 stock_list 表）",
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
