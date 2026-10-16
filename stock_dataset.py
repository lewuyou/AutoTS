# -*- coding: utf-8 -*-
"""模型训练前的数据表生成：替换 AutoTS 内置数据源，从 download/autots.duckdb 读取。

按总市值降序逐股构建 AutoTS 宽表（DatetimeIndex 行 × 每列一条序列）。完整列清单：

    date（索引）              交易日期，仅含该股交易日
    tgt_ret_d{H}              预测目标（首列）：H 个交易日后的收益率 = 未来后复权收盘价 /
                              当日后复权收盘价 - 1，H 由 TARGET_HORIZON / --horizon 设置，默认 7。
                              末尾 H 行因无未来数据为 NaN，已剔除
    px_open/px_close/px_high/px_low   后复权开/收/高/低价（元），源：腾讯 akshare_tx
    px_turnover               换手率（小数，0.05 = 5%），停牌日为 NaN（不填充）
    baidu_search_all          百度指数·搜索整体（PC+移动，整数），关键词 = 股票中文名
    baidu_feed                百度指数·资讯指数（2017-07-03 起才有）
    val_pe/val_pb/val_ps/val_pcf      东财估值 TTM：市盈率/市净率/市销率/市现率
    rzrq_fin_tval_ratio       融资净买入占成交额比（%）
    rzrq_loan_tval_ratio      融券净卖出占成交额比（%）
    macro_close_*             TradingView 宏观品种收盘价（股指/商品期货/债券收益率/汇率），
                              列名由品种代码清洗而来（'COMEX:GC1!' -> macro_close_GC1），
                              列数随库中品种而定，当前含 GC1 黄金、HG1 铜、CL1 原油、NG1 天然气、
                              ATW1 小麦、SPX 标普、IXIC 纳指、NDX 纳指100、VIX、SOX 费城半导体、
                              HSI 恒指、000001 上证、510300 沪深300ETF、CN10Y/US10Y 国债收益率、
                              DXY 美元指数、USDCNY 汇率

对齐规则：以该股 akshare_tx 交易日历为骨架（行只含交易日），辅助数据左连接后前向填充
（上限 FILL_LIMIT 个交易日），避免周末/节假日的百度指数、海外宏观数据产生错位行。

用法：
    python stock_dataset.py                 # 演示：构建市值最大股票的宽表并打印概况
    python stock_dataset.py --symbol 601398 # 指定股票
    python stock_dataset.py --top 10        # 打印市值前 10 股票各自宽表概况
    python stock_dataset.py --horizon 5     # 预测目标改为 5 个交易日后的收益率

    from stock_dataset import iter_stock_frames
    for symbol, name, total_mv, df in iter_stock_frames():
        model.fit(df)  # AutoTS 宽表，首列 tgt_ret_d{H} 为预测目标

导出 CSV：加 --csv 后，每只股票宽表另存为 data/{symbol}_{name}.csv
（utf-8-sig 编码，Excel 直接打开中文不乱码），目录可用 --out-dir 指定。
"""

import argparse
import os
import re
import sys

import duckdb
import numpy as np
import pandas as pd

DEFAULT_DB_PATH = "download/autots.duckdb"
DEFAULT_CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

ADJUST = "hfq"  # 库中 akshare_tx 仅存有后复权数据
TARGET_HORIZON = 7  # 预测目标：未来第 N 个交易日的收益率
FILL_LIMIT = 5  # 辅助序列前向填充上限（交易日），超过即视为真实缺失
MIN_HISTORY_DAYS = 250  # 交易日少于此数的股票跳过（约一年，不足以支撑训练/验证切分）

PRICE_COLUMNS = ["open", "close", "high", "low", "turnover"]
GUZHI_INDICATORS = ["pe", "pb", "ps", "pcf"]
BAIDU_SOURCES = ["search_all", "feed"]
RZRQ_COLUMNS = ["fin_tval_ratio", "loan_tval_ratio"]


def target_column_name(horizon=TARGET_HORIZON):
    return f"tgt_ret_d{horizon}"


def connect(db_path=DEFAULT_DB_PATH):
    return duckdb.connect(db_path, read_only=True)


def load_stock_universe(con, min_market_cap=0.0):
    """股票列表按总市值降序；剔除无市值快照的个股。"""
    return con.execute(
        """
        SELECT symbol, name, total_mv
        FROM stock_list
        WHERE total_mv IS NOT NULL AND total_mv > ?
        ORDER BY total_mv DESC
        """,
        [min_market_cap],
    ).fetchdf()


def _clean_macro_name(symbol):
    """'COMEX:GC1!' -> 'GC1'；'SSE:000001' -> 'SSE000001'（避免与纯数字列名混淆）。"""
    tail = symbol.split(":")[-1].rstrip("!")
    return re.sub(r"[^A-Za-z0-9]", "", tail)


def load_macro_closes(con):
    """宏观品种收盘价宽表：tradingview_macro 全部品种（股指/商品/债券收益率/汇率）。"""
    df = con.execute(
        "SELECT symbol, date, close FROM tradingview_macro WHERE close IS NOT NULL"
    ).fetchdf()
    if df.empty:
        return pd.DataFrame()
    wide = df.pivot(index="date", columns="symbol", values="close").sort_index()
    wide.columns = [f"macro_close_{_clean_macro_name(c)}" for c in wide.columns]
    wide.index = pd.DatetimeIndex(wide.index)
    return wide


def load_price_frame(con, symbol, target_horizon=TARGET_HORIZON):
    """后复权行情 + 目标列（未来第 target_horizon 个交易日的收益率），交易日索引。"""
    df = con.execute(
        """
        SELECT date, open, close, high, low, turnover
        FROM akshare_tx
        WHERE symbol = ? AND adjust = ?
        ORDER BY date
        """,
        [symbol, ADJUST],
    ).fetchdf()
    if df.empty:
        return df
    df["date"] = pd.DatetimeIndex(df["date"])
    df = df.set_index("date")
    df = df.rename(columns={c: f"px_{c}" for c in PRICE_COLUMNS})
    # shift 为负取未来收盘价，换算为区间收益率：末尾 target_horizon 行无未来数据为 NaN，由 build_stock_frame 剔除
    future_close = df["px_close"].shift(-target_horizon)
    df.insert(0, target_column_name(target_horizon), future_close / df["px_close"] - 1)
    return df


def load_baidu_frame(con, keyword):
    """百度指数 pivot：search_all=搜索整体（PC+移动），feed=资讯指数（2017-07-03 起）。"""
    df = con.execute(
        "SELECT date, source, value FROM baidu WHERE keyword = ? ORDER BY date",
        [keyword],
    ).fetchdf()
    if df.empty:
        return pd.DataFrame()
    wide = df.pivot(index="date", columns="source", values="value").sort_index()
    wide = wide.reindex(columns=[c for c in BAIDU_SOURCES if c in wide.columns])
    wide.columns = [f"baidu_{c}" for c in wide.columns]
    wide.index = pd.DatetimeIndex(wide.index)
    return wide


def load_guzhi_frame(con, symbol):
    """东财估值 TTM pivot：pe/pb/ps/pcf。"""
    df = con.execute(
        "SELECT date, indicator, value FROM guzhi WHERE symbol = ? ORDER BY date",
        [symbol],
    ).fetchdf()
    if df.empty:
        return pd.DataFrame()
    wide = df.pivot(index="date", columns="indicator", values="value").sort_index()
    wide = wide.reindex(columns=[c for c in GUZHI_INDICATORS if c in wide.columns])
    wide.columns = [f"val_{c}" for c in wide.columns]
    wide.index = pd.DatetimeIndex(wide.index)
    return wide


def load_rzrq_frame(con, symbol):
    """两融：融资净买入/融券净卖出占成交额比（%）。"""
    df = con.execute(
        "SELECT date, fin_tval_ratio, loan_tval_ratio FROM rzrq WHERE code = ? ORDER BY date",
        [symbol],
    ).fetchdf()
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.DatetimeIndex(df["date"])
    return df.set_index("date").rename(columns={c: f"rzrq_{c}" for c in RZRQ_COLUMNS})


def build_stock_frame(con, symbol, name, macro_closes=None, min_history_days=MIN_HISTORY_DAYS,
                      target_horizon=TARGET_HORIZON):
    """单股训练宽表：交易日骨架 + 目标收益率 + 全部辅助列（前向填充对齐）。

    返回 None 表示历史长度不足，调用方应跳过。
    """
    prices = load_price_frame(con, symbol, target_horizon)
    if prices.empty or len(prices) < min_history_days:
        return None

    if macro_closes is None:
        macro_closes = load_macro_closes(con)

    aux_frames = [
        load_baidu_frame(con, name),
        load_guzhi_frame(con, symbol),
        load_rzrq_frame(con, symbol),
        macro_closes,
    ]
    aux = pd.concat([f for f in aux_frames if not f.empty], axis=1, sort=False)

    # 以交易日为骨架左连接，辅助列前向填充；行情/目标列保持原样（缺失即真实停牌/无未来数据）
    frame = prices.join(aux.reindex(prices.index).ffill(limit=FILL_LIMIT))
    return frame.dropna(subset=[target_column_name(target_horizon)])


def iter_stock_frames(db_path=DEFAULT_DB_PATH, min_market_cap=0.0,
                      min_history_days=MIN_HISTORY_DAYS, max_stocks=None,
                      target_horizon=TARGET_HORIZON):
    """按总市值降序逐股产出 (symbol, name, total_mv, frame)；历史不足的跳过。"""
    con = connect(db_path)
    try:
        universe = load_stock_universe(con, min_market_cap)
        macro_closes = load_macro_closes(con)
        emitted = 0
        for row in universe.itertuples(index=False):
            frame = build_stock_frame(
                con, row.symbol, row.name,
                macro_closes=macro_closes,
                min_history_days=min_history_days,
                target_horizon=target_horizon,
            )
            if frame is None:
                continue
            yield row.symbol, row.name, row.total_mv, frame
            emitted += 1
            if max_stocks is not None and emitted >= max_stocks:
                return
    finally:
        con.close()


def save_frame_csv(frame, symbol, name, out_dir=DEFAULT_CSV_DIR):
    """宽表存 CSV。utf-8-sig 带 BOM，保证 Excel 打开中文列名/文件名内容不乱码。"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{symbol}_{name}.csv")
    frame.to_csv(path, index_label="date", encoding="utf-8-sig")
    return path


def describe_frame(symbol, name, total_mv, frame):
    non_null = frame.notna().sum()
    print(f"\n=== {symbol} {name}  总市值 {total_mv / 1e8:.0f} 亿 ===")
    print(f"区间 {frame.index.min().date()} ~ {frame.index.max().date()}  共 {len(frame)} 个交易日")
    for col in frame.columns:
        print(f"  {col:24s} 非空 {non_null[col]:5d} / {len(frame)}")


def main():
    parser = argparse.ArgumentParser(description="生成 AutoTS 训练用单股宽表（数据源：download/autots.duckdb）")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="DuckDB 路径")
    parser.add_argument("--symbol", default=None, help="指定单只股票代码（6 位），默认取市值最大")
    parser.add_argument("--top", type=int, default=1, help="打印市值前 N 只股票的宽表概况")
    parser.add_argument("--horizon", type=int, default=TARGET_HORIZON,
                        help="预测目标：未来第 N 个交易日的收益率（默认 7）")
    parser.add_argument("--csv", action="store_true", help="将宽表导出为 CSV（默认存到脚本目录下 data/）")
    parser.add_argument("--out-dir", default=DEFAULT_CSV_DIR, help="CSV 输出目录")
    args = parser.parse_args()

    con = connect(args.db)
    try:
        macro_closes = load_macro_closes(con)
        if args.symbol:
            row = con.execute(
                "SELECT symbol, name, total_mv FROM stock_list WHERE symbol = ?", [args.symbol]
            ).fetchone()
            if row is None:
                sys.exit(f"stock_list 中找不到 {args.symbol}")
            candidates = [row]
        else:
            candidates = load_stock_universe(con).head(args.top).itertuples(index=False)

        for row in candidates:
            frame = build_stock_frame(con, row[0], row[1], macro_closes=macro_closes,
                                      target_horizon=args.horizon)
            if frame is None:
                print(f"{row[0]} {row[1]} 历史不足 {MIN_HISTORY_DAYS} 个交易日，跳过")
                continue
            describe_frame(row[0], row[1], row[2], frame)
            print(frame.tail(3).to_string())
            if args.csv:
                print(f"已导出 {save_frame_csv(frame, row[0], row[1], args.out_dir)}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
