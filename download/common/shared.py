# -*- coding: utf-8 -*-
"""各数据源共享的领域级小工具：stock_list 读取、ST 过滤、增量水位、水位分布日志、
代码补市场前缀、交易日历。

原来 akshare/baidu 各自维护一份近似的实现（load_stock_list、
get_max_dates/load_last_dates、log_db_state、ST 名称判断、code_to_symbol），
此处统一，各数据源只保留"按自身表结构整理结果"的那一层。
"""

import datetime

from collections import Counter

from download.common import storage
from download.common.paths import TEMP_DIR


def load_stock_list(db_path):
    """从 stock_list 表读全部股票 (symbol, list_date, name) 三元组，按总市值 total_mv 从大到小排序。

    市值缺失的排最后，市值相同按 symbol 排序；list_date 缺失用 1990-01-01 兜底，name 缺失用空串。
    """
    con = storage.connect(db_path, read_only=True)
    try:
        rows = con.execute(
            "SELECT symbol, COALESCE(list_date, DATE '1990-01-01'), COALESCE(name, '') "
            "FROM stock_list ORDER BY total_mv DESC NULLS LAST, symbol"
        ).fetchall()
    finally:
        con.close()
    return [(str(s), d, (n or "")) for s, d, n in rows]


def load_stock_names_by_total_mv(db_path):
    """从 stock_list 表读取去重的非空股票简称，按总市值从大到小排序（同名股票取市值最大者）。"""
    return list(dict.fromkeys(n for _, _, n in load_stock_list(db_path) if n))


def code_to_symbol(code):
    """A 股代码 -> 带市场前缀的 symbol。6/9 开头上交所 sh，0/2/3 开头深交所 sz，4/8 开头北交所 bj。"""
    code = str(code).strip()
    if code.startswith(("sh", "sz", "bj")):
        return code
    if code[0] in ("6", "9"):
        return f"sh{code}"
    if code[0] in ("0", "2", "3"):
        return f"sz{code}"
    if code[0] in ("4", "8"):
        return f"bj{code}"
    return f"sz{code}"


def is_st_name(name):
    """判断股票名称是否含 ST（含 *ST）。"""
    return "ST" in str(name).upper()


def load_max_dates(db_path, source_name, table, key_cols, date_col="date",
                   where_clause="", where_params=(), stage_path=None):
    """主库 + 未合并暂存库按 key_cols 分组取 date_col 最大水位。

    返回 [(key_val1, ..., max_date), ...]，各数据源自行整理成自身需要的 dict。
    """
    return storage.max_date_by_keys_merged(
        db_path, storage.pending_stage_files(source_name, TEMP_DIR, stage_path),
        table, key_cols, date_col, where_clause, where_params)


def load_max_date_map(db_path, source_name, table, key_col, date_col="date",
                      where_clause="", where_params=(), stage_path=None, as_string=False,
                      log=None, unit="条"):
    """单键表增量水位整理成 {key: max_date}。

    复用 load_max_dates（主库 + 未合并暂存库合并水位），把 [(key, date), ...]
    转成 dict；as_string=True 时日期转 'YYYY-MM-DD' 字符串，便于与抓取回的字符串日期比较。
    log 给定时打印库内最后日期分布（统一输出）。
    """
    rows = load_max_dates(db_path, source_name, table, [key_col], date_col,
                          where_clause, where_params, stage_path)
    if log is not None:
        log_max_date_distribution(log, {str(r[0]): r[1] for r in rows}, table, unit=unit)
    return {
        str(r[0]): (r[1].isoformat() if as_string else r[1])
        for r in rows
    }


def _as_date(v):
    """兼容 datetime.date 与 'YYYY-MM-DD' 字符串，返回 datetime.date。"""
    if isinstance(v, datetime.date):
        return v
    return datetime.date.fromisoformat(str(v)[:10])


def _log_date_distribution(log, dates, table_label, unit):
    """打印最后日期分布：最新日期 + 各日期计数（dates 为 date 或 'YYYY-MM-DD' 字符串序列）。"""
    dates = [_as_date(d) for d in dates]
    dist = Counter(d.strftime("%Y-%m-%d") for d in dates)
    latest = max(dates)
    log.info("主库 %s 已有 %d %s，最新日期 %s",
             table_label, len(dates), unit, latest.strftime("%Y-%m-%d"))
    log.info("各序列最后日期分布（前 10 个日期）：")
    for d, c in sorted(dist.items(), reverse=True)[:10]:
        log.info("    %s: %d %s", d, c, unit)


def log_max_date_distribution(log, max_dates, table_label, unit="条"):
    """打印库内各序列最后日期分布，便于看出哪些序列滞后；空水位打印全量提示。

    max_dates 形如 {key: datetime.date 或 'YYYY-MM-DD' 字符串}；unit 为计数单位词（如 "只股票"）。
    """
    if not max_dates:
        log.info("主库 %s 无记录，本次按首次全量路径抓取", table_label)
        return
    _log_date_distribution(log, max_dates.values(), table_label, unit)


def log_last_dates_distribution(log, last_dates, table_label, unit="条"):
    """打印多源水位分布；last_dates 形如 {key: {subkey: date}}（如 baidu 的
    {keyword: {search_all: date, feed: date}}）。每个 (key, subkey) 计为一条序列。
    """
    if not last_dates:
        log.info("主库 %s 无记录，本次按首次全量路径抓取", table_label)
        return
    dates = [d for srcs in last_dates.values() for d in srcs.values()]
    _log_date_distribution(log, dates, table_label, unit)


MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 5  # A股 15:00 收盘，留 5 分钟缓冲


def last_completed_trading_day(db_path, end_date, now=None):
    """主库 holiday_calendar 里最近一个"数据应已齐全"的交易日，用于请求前跳过已最新股票。

    取 <= end_date 的交易日中最大的；若该日就是今天且尚未收盘，则回退到再前一个交易日
    （盘中当日数据不齐，不视为已齐全）。日历表缺失返回 None，调用方回退到逐只请求。
    """
    con = storage.connect(db_path, read_only=True)
    try:
        if not storage.table_exists(con, "holiday_calendar"):
            return None
        rows = con.execute(
            "SELECT date FROM holiday_calendar "
            "WHERE is_holiday = FALSE AND date <= ? ORDER BY date DESC LIMIT 2",
            [end_date],
        ).fetchall()
    finally:
        con.close()
    if not rows:
        return None
    latest = rows[0][0]
    now = now or datetime.datetime.now()
    if latest == end_date and len(rows) > 1 and (now.hour, now.minute) < (MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE):
        return rows[1][0]
    return latest
