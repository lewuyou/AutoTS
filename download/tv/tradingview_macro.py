# -*- coding: utf-8 -*-
"""TradingView 宏观经济数据抓取入库模块（OHLC + 成交量）。

品种清单见同目录 macro_config.py（国债收益率、股指、汇率、商品期货等）。
复用 tradingview.py 的 WebSocket 抓取与 DuckDB 入库逻辑，入库到 tradingview_macro 表。

更新模式：
    默认增量：按库内每个品种已有最大日期续抓。注意 TradingView WebSocket 单次
              始终返回全历史 5000 根 K 线，"增量"只是本地按 [库内最大日期, 今天]
              过滤后 UPSERT（重复日期自动覆盖），因此漏跑多久都能一次补齐，
              网络开销与全量相同，省的只是入库行数。
    --reset：先清空 tradingview_macro 表再全量抓取（一次性初始化用）。

用法（独立运行，需在项目根目录）：
    python -m download.tv.tradingview_macro            # 增量更新全部宏观品种
    python -m download.tv.tradingview_macro --reset    # 清表后全量抓取
    python -m download.tv.tradingview_macro --symbols "TVC:CN10Y,USDCNY"
    python -m download.tv.tradingview_macro --no-proxy

依赖：
    pip install duckdb websocket-client

入库表 tradingview_macro 字段含义：
    symbol  TradingView 品种代码（如 "TVC:CN10Y"、"USDCNY"）
    name    中文名称（如 "中国10年期国债收益率"）
    date    交易日（由 UTC 时间戳转北京时间日期）
    open/high/low/close  开高低收（收益率类为百分比数值，汇率/指数为点位，期货为合约价）
    volume  成交量（股/手；指数、收益率、汇率类品种通常为空）
"""

import argparse
import datetime
import os
import sys

from . import tradingview
from .macro_config import MACRO_SYMBOLS

TABLE_NAME = "tradingview_macro"
DEFAULT_DB_PATH = tradingview.DEFAULT_DB_PATH


def ingest(rows, db_path=DEFAULT_DB_PATH):
    """UPSERT 到 DuckDB 的 tradingview_macro 表。"""
    tradingview._require_duckdb()
    import duckdb

    con = duckdb.connect(db_path)
    try:
        con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                symbol TEXT,
                name TEXT,
                date DATE,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                PRIMARY KEY (symbol, date)
            )
            """
        )
        con.executemany(
            f"INSERT OR REPLACE INTO {TABLE_NAME} (symbol, name, date, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    finally:
        con.close()
    return len(rows)


def reset_table(db_path=DEFAULT_DB_PATH):
    """清空 tradingview_macro 表（一次性初始化用）。"""
    tradingview._require_duckdb()
    import duckdb

    con = duckdb.connect(db_path)
    try:
        con.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
    finally:
        con.close()


def get_last_dates(db_path=DEFAULT_DB_PATH):
    """返回 {symbol: 库内最大日期(YYYY-MM-DD)}，表不存在时返回空 dict。"""
    tradingview._require_duckdb()
    import duckdb

    if not os.path.exists(db_path):
        return {}
    con = duckdb.connect(db_path, read_only=True)
    try:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if TABLE_NAME not in tables:
            return {}
        return {
            sym: d.isoformat()
            for sym, d in con.execute(
                f"SELECT symbol, MAX(date) FROM {TABLE_NAME} GROUP BY symbol"
            ).fetchall()
        }
    finally:
        con.close()


def run(symbols=None, start=None, end=None, reset=False, db_path=None,
        interval="1D", use_proxy=True):
    """供外部调用的入口。

    参数：
        symbols: TradingView 品种代码列表（或逗号分隔串），默认 macro_config.MACRO_SYMBOLS 全部
        start: 起始日期 YYYY-MM-DD；None 时增量模式取库内最大日期，reset 模式抓全历史
        end: 结束日期 YYYY-MM-DD，默认今天
        reset: True 则先清空 tradingview_macro 表再全量抓取
        db_path: DuckDB 文件路径，默认 download/autots.duckdb
        interval: K 线周期，默认 1D
        use_proxy: 是否走本机 7897 代理（默认 True）
    返回：
        dict 包含 rows_count, db_path, series, failed 等
    """
    if symbols is None:
        entries = list(MACRO_SYMBOLS)
    else:
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(",") if s.strip()]
        name_map = dict(MACRO_SYMBOLS)
        entries = [(s, name_map.get(s, "")) for s in symbols]
    if not entries:
        raise ValueError("未提供品种代码")

    db_path = db_path or DEFAULT_DB_PATH
    end = end or datetime.date.today().isoformat()

    if reset:
        print(f"[TradingView宏观] 清空表 {TABLE_NAME} ({db_path})")
        reset_table(db_path)

    last_dates = {} if reset else get_last_dates(db_path)

    total_rows = 0
    series = {}
    failed = {}

    for symbol, name in entries:
        # 增量：从库内最大日期起（当日数据可能未收盘，重抓覆盖最后一天）
        eff_start = start if start is not None else last_dates.get(symbol)
        label = f"{symbol} {name}".strip()
        print(f"[TradingView宏观] 抓取 {label} {interval} start={eff_start or '全历史'} ...")
        try:
            bars = tradingview.fetch_bars(symbol, interval=interval, use_proxy=use_proxy)
        except Exception as e:
            failed[symbol] = str(e)
            print(f"[TradingView宏观] {label} 失败: {e}", file=sys.stderr)
            continue
        base_rows = tradingview.bars_to_rows(symbol, bars, start=eff_start, end=end)
        rows = [(sym, name, d, o, h, l, c, v) for (sym, d, o, h, l, c, v) in base_rows]
        if not rows:
            series[symbol] = {"n": 0, "up_to_date": True}
            print(f"[TradingView宏观] {label} 已是最新，无新增")
            continue
        n = ingest(rows, db_path=db_path)
        total_rows += n
        series[symbol] = {
            "name": name,
            "n": len(rows),
            "first": rows[0][2],
            "last": rows[-1][2],
        }
        print(f"[TradingView宏观] {label} 入库 {n} 行  {rows[0][2]} ~ {rows[-1][2]}")

    print("\n[TradingView宏观] 入库概览")
    for symbol, s in series.items():
        if s.get("up_to_date"):
            print(f"  {symbol}: 已是最新")
        else:
            print(f"  {symbol}: n={s['n']:5d}  {s['first']} ~ {s['last']}  {s.get('name', '')}")

    result = {
        "rows_count": total_rows,
        "db_path": db_path,
        "series": series,
    }
    if failed:
        result["failed"] = failed
        if not series:
            raise RuntimeError(f"全部失败: {failed}")
    return result


def main():
    parser = argparse.ArgumentParser(description="TradingView 宏观经济数据抓取入库（增量）")
    parser.add_argument("--symbols", default=None,
                        help="逗号分隔 TradingView 品种代码，默认 macro_config 全部")
    parser.add_argument("--start", default=None, help="起始日期 YYYY-MM-DD（默认增量取库内最大日期）")
    parser.add_argument("--end", default=None, help="结束日期 YYYY-MM-DD（默认今天）")
    parser.add_argument("--reset", action="store_true", help="清空表后全量抓取（一次性初始化）")
    parser.add_argument("--interval", default="1D", help="K 线周期，默认 1D")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="DuckDB 文件路径")
    parser.add_argument("--no-proxy", action="store_true", help="不走本机 7897 代理")
    args = parser.parse_args()

    run(
        symbols=args.symbols,
        start=args.start,
        end=args.end,
        reset=args.reset,
        db_path=args.db,
        interval=args.interval,
        use_proxy=not args.no_proxy,
    )


if __name__ == "__main__":
    main()
