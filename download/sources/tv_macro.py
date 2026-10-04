# -*- coding: utf-8 -*-
"""TradingView 宏观经济数据抓取（接入模块，日频 OHLC + 成交量）。

仅保留数据源特有逻辑：TradingView WebSocket 逐品种拉 K 线（复用 download.tv.tradingview
的 fetch_bars/bars_to_rows）、品种清单读写 tradingview_macro_symbols 配置表。
日志/建表/UPSERT/暂存路径/水位/合并/结果统一走 download.common。

品种清单：以主库 tradingview_macro_symbols 表为准（直接改表即可维护）；
表不存在或为空时用 download/tv/macro_config.py 的 MACRO_SYMBOLS 初始化；
macro_config.py 更新后执行 --sync-symbols 可重新同步进表（只增/改不删）。

增量规则：按库内每个品种最大日期续抓（含水位日当天，重抓覆盖，UPSERT 幂等）。
注意 TradingView WebSocket 单次始终返回全历史 5000 根 K 线，"增量"只是本地按
[库内最大日期, 今天] 过滤后 UPSERT，漏跑多久都能一次补齐，网络开销与全量相同。

用法（独立运行）：
    python -m download.sources.tv_macro                       # 增量更新全部宏观品种
    python -m download.sources.tv_macro --symbols "TVC:CN10Y,USDCNY"
    python -m download.sources.tv_macro --sync-symbols        # 把 macro_config 清单同步进配置表后再抓
    python -m download.sources.tv_macro --reset               # 忽略水位全量重抓（UPSERT，不清表）
    python -m download.sources.tv_macro --stage auto          # 暂存模式
    python -m download.sources.tv_macro --merge "download/temp/tv_macro_stage_*.duckdb"

依赖：
    pip install duckdb websocket-client

入库表 tradingview_macro 字段含义：
    symbol  TradingView 品种代码（如 "TVC:CN10Y"、"USDCNY"）
    name    中文名称（如 "中国10年期国债收益率"）
    date    交易日（由 UTC 时间戳转北京时间日期）
    open/high/low/close  开高低收（收益率类为百分比数值，汇率/指数为点位，期货为合约价）
    volume  成交量（股/手；指数、收益率、汇率类品种通常为空）

配置表 tradingview_macro_symbols 字段含义（品种清单，主键 symbol；首次运行由 macro_config.py 初始化）：
    symbol  TradingView 品种代码（如 "TVC:CN10Y"、"USDCNY"）
    name    中文名称（如 "中国10年期国债收益率"）
"""

import datetime
import os

from download.common import cli
from download.common import logging as common_logging
from download.common import results
from download.common import shared
from download.common import storage
from download.common.paths import DEFAULT_DB_PATH, TEMP_DIR
from download.tv import tradingview
from download.tv.macro_config import MACRO_SYMBOLS

SOURCE_NAME = "tv_macro"
TABLE_NAME = "tradingview_macro"
TABLE_SCHEMA = """(
    symbol TEXT,
    name TEXT,
    date DATE,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume DOUBLE,
    PRIMARY KEY (symbol, date)
)"""
COLUMNS = ["symbol", "name", "date", "open", "high", "low", "close", "volume"]

SYMBOLS_TABLE = "tradingview_macro_symbols"
SYMBOLS_SCHEMA = """(
    symbol TEXT,
    name TEXT,
    PRIMARY KEY (symbol)
)"""
SYMBOLS_COLUMNS = ["symbol", "name"]


def sync_symbols(db_path, log=None):
    """把 macro_config.MACRO_SYMBOLS UPSERT 进品种配置表，返回写入行数。

    只增/改不删：表中手动加入的品种保留。
    """
    n = storage.ingest(db_path, SYMBOLS_TABLE, SYMBOLS_SCHEMA, SYMBOLS_COLUMNS,
                       list(MACRO_SYMBOLS))
    if log:
        log.info("品种配置表 %s 已从 macro_config 同步 %d 个品种", SYMBOLS_TABLE, n)
    return n


def load_symbols(db_path):
    """读品种配置表，返回 [(symbol, name)]；库/表不存在或为空返回 []。"""
    if not os.path.exists(db_path):
        return []
    con = storage.connect(db_path, read_only=True)
    try:
        if not storage.table_exists(con, SYMBOLS_TABLE):
            return []
        return [(s, n or "") for s, n in con.execute(
            f"SELECT symbol, name FROM {SYMBOLS_TABLE} ORDER BY symbol").fetchall()]
    finally:
        con.close()


def resolve_entries(symbols, table_entries, db_path, log):
    """确定本次抓取的 [(symbol, 中文名称)]。

    symbols 为 None 时以配置表为准，表为空则用 macro_config 初始化；
    显式传入 symbols 时名称查配置表，查不到再查 macro_config，都没有补空串。
    """
    if symbols is None:
        if table_entries:
            return table_entries
        log.info("品种配置表为空，用 macro_config 初始化（%d 个品种）", len(MACRO_SYMBOLS))
        sync_symbols(db_path, log)
        return list(MACRO_SYMBOLS)
    if isinstance(symbols, str):
        symbols = [s.strip() for s in symbols.split(",") if s.strip()]
    name_map = dict(MACRO_SYMBOLS) | dict(table_entries)
    return [(s, name_map.get(s, "")) for s in symbols]


merge_stages = storage.make_merge_stages(TABLE_NAME, TABLE_SCHEMA, COLUMNS, DEFAULT_DB_PATH)


def run(symbols=None, start=None, end=None, interval="1D", use_proxy=True, reset=False,
        sync_symbols_first=False, db_path=None, stage_path=None, run_id=""):
    """执行宏观品种抓取，写入暂存库（或主库），不在此合并。

    参数：
        symbols: TradingView 品种代码列表（或逗号分隔串），默认读 tradingview_macro_symbols 配置表
        start: 起始日期 YYYY-MM-DD；None 时按库内水位增量（reset 时忽略水位抓全历史）
        end: 统一截止日期 YYYY-MM-DD，默认今天
        interval: K 线周期，默认 1D
        use_proxy: 是否走本机 7897 代理（默认 True，TradingView 需代理）
        reset: True 忽略库内水位全量重抓（UPSERT 幂等，不清表）
        sync_symbols_first: True 抓取前先把 macro_config 清单 UPSERT 进品种配置表
        db_path: 主库路径，默认 download/autots.duckdb
        stage_path: 暂存 DuckDB 路径；"auto" 自动生成，None 直入主库；增量起点读主库+暂存库水位
    返回：
        RunResult（fetch_status/rows_staged/failed 等）
    """
    log = common_logging.get_logger(SOURCE_NAME, run_id)
    db_path = db_path or DEFAULT_DB_PATH
    res = results.RunResult(SOURCE_NAME, run_id)

    end = str(end).strip().replace("/", "-") if end else datetime.date.today().isoformat()

    stage_path = storage.make_stage_path(SOURCE_NAME, TEMP_DIR, stage_path)
    if stage_path:
        log.info("暂存模式：写入 %s，事后用 merge 合并入主库", stage_path)
    res.stage_path = stage_path
    ingest_db = stage_path or db_path

    if sync_symbols_first:
        sync_symbols(db_path, log)
    entries = resolve_entries(symbols, load_symbols(db_path), db_path, log)
    if not entries:
        raise ValueError("未提供品种代码")

    # 单键水位：主库 + 未合并暂存库合并取每个品种最大日期
    if reset or start is not None:
        max_dates = {}
        if reset:
            log.info("reset 模式：忽略库内水位，全历史重抓（UPSERT 覆盖）")
    else:
        max_dates = shared.load_max_date_map(
            db_path, SOURCE_NAME, TABLE_NAME, "symbol", as_string=True, stage_path=stage_path,
            log=log, unit="个品种",
        )

    total_rows = 0
    series = {}
    failed = {}
    no_new = 0

    for idx, (symbol, name) in enumerate(entries, 1):
        # 增量起点含水位日当天（当日数据可能未收盘，重抓覆盖最后一天）
        eff_start = start if start is not None else max_dates.get(symbol)
        label = f"{symbol} {name}".strip()
        log.info("[%d/%d] 抓取 %s %s start=%s ...",
                 idx, len(entries), label, interval, eff_start or "全历史")
        try:
            bars = tradingview.fetch_bars(symbol, interval=interval, use_proxy=use_proxy)
        except Exception as e:
            failed[symbol] = str(e)
            log.error("%s 失败: %s", label, e)
            continue
        base_rows = tradingview.bars_to_rows(symbol, bars, start=eff_start, end=end)
        rows = [(sym, name, d, o, h, l, c, v) for (sym, d, o, h, l, c, v) in base_rows]
        if not rows:
            # 库中已有该品种数据且无新交易日 = 已最新，不算失败
            if symbol in max_dates:
                no_new += 1
                log.info("%s 已是最新，无新增", label)
                continue
            failed[symbol] = "无数据"
            log.warning("%s 无数据", label)
            continue
        n = storage.ingest(ingest_db, TABLE_NAME, TABLE_SCHEMA, COLUMNS, rows)
        total_rows += n
        first, last = rows[0][2], rows[-1][2]
        series[symbol] = {"name": name, "n": n, "first": first, "last": last}
        log.info("%s 入库 %d 行 %s ~ %s", label, n, first, last)

    if failed:
        res.failed_path = results.failed_path_for(SOURCE_NAME, TEMP_DIR)
        results.write_failed_list(failed, res.failed_path)
        log.warning("%d 个品种失败，清单: %s", len(failed), res.failed_path)

    log.info("完成：成功 %d 个，失败 %d 个，无新数据 %d 个，累计入库 %d 行",
             len(series), len(failed), no_new, total_rows)

    res.rows_staged = total_rows
    res.success = len(series)
    res.failed = len(failed)
    res.no_new = no_new
    res.detail = {"series": series}
    if failed:
        if not series and not no_new:
            res.fetch_status = results.FETCH_FAILED
            res.error = f"全部失败: {failed}"
        else:
            res.fetch_status = results.FETCH_PARTIAL
    else:
        res.fetch_status = results.FETCH_OK
    return res.finish()


def _add_args(parser):
    parser.add_argument("--symbols", default=None,
                        help="逗号分隔 TradingView 品种代码，默认读 tradingview_macro_symbols 配置表")
    parser.add_argument("--start", default=None, help="起始日期 YYYY-MM-DD（默认按库内水位增量）")
    parser.add_argument("--end", default=None, help="截止日期 YYYY-MM-DD（默认今天）")
    parser.add_argument("--interval", default="1D", help="K 线周期，默认 1D")
    parser.add_argument("--no-proxy", action="store_true", help="不走本机 7897 代理")
    parser.add_argument("--reset", action="store_true",
                        help="忽略库内水位全量重抓（UPSERT 幂等，不清表）")
    parser.add_argument("--sync-symbols", action="store_true",
                        help="抓取前先把 macro_config 清单 UPSERT 进品种配置表（只增/改不删）")
    parser.add_argument("--stage", default=None, help="暂存 DuckDB 路径；auto 自动生成，不设则直入主库")


def _build_kwargs(args, db_path):
    return dict(
        symbols=args.symbols,
        start=args.start,
        end=args.end,
        interval=args.interval,
        use_proxy=not args.no_proxy,
        reset=args.reset,
        sync_symbols_first=args.sync_symbols,
        db_path=db_path,
        stage_path=args.stage,
    )


def _log_start(log, args, db_path):
    log.info("开始，主库: %s，品种: %s，周期: %s，范围: %s ~ %s，代理: %s，reset: %s，sync: %s",
             db_path, args.symbols or "配置表", args.interval,
             args.start or "库内水位", args.end or "今天", not args.no_proxy, args.reset,
             args.sync_symbols)


def main():
    cli.standard_main(
        source_name=SOURCE_NAME,
        description="TradingView 宏观经济数据抓取入库（日频 OHLC+成交量，入 tradingview_macro 表）",
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
