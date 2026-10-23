# -*- coding: utf-8 -*-
"""行业维度训练数据准备：生成 AutoTS 可直接 fit 的行业收盘价宽表 + 宏观成交量回归器。

从 download/autots.duckdb（只读）读取：

    stock_list.industry_l2        行业分类（申万/东财二级），圈定行业内全部 A 股
    akshare_tx.close              个股后复权收盘价（元），adjust='hfq'，源：腾讯行情
    tradingview_macro.close/volume 宏观品种收盘价与成交量（TradingView 原始数值；
                                  收盘价为点位/元，ETF 成交量为股数，指数成交量
                                  为 TradingView 汇总口径，均原样使用不换算）：
                                      SSE:000001  上证综合指数
                                      SSE:510300  沪深300ETF（510300）
                                      SPCFD:SPX   标普500指数

输出两张宽表（index 均为 pd.DatetimeIndex，仅含 A 股交易日，二者行完全对齐）：

  1. 收盘价表（AutoTS 的 df，长/宽表二选一，这里用宽表）：
     date（索引）              交易日
     {symbol}_{name}           该股后复权收盘价（元），列名 = 6 位代码_简称，
                               如 600519_贵州茅台；停牌日为 NaN（保留，AutoTS
                               在模型层 fillna 处理）

  2. 回归器表（AutoTS 的 future_regressor）：
     date（索引）              交易日
     close_上证综指            上证综合指数收盘点位
     close_沪深300ETF          沪深300ETF 收盘价（元）
     close_标普500             标普500指数收盘点位
     vol_上证综指              上证综合指数成交量
     vol_沪深300ETF            沪深300ETF 成交量（股）
     vol_标普500               标普500指数成交量

处理规则：
    * 交易日历 = 行业内全部股票 akshare_tx 日期并集（≈A 股交易日历，天然剔除
      周末/节假日，避免 AutoTS 的 df_cleanup 按日历日 resample 插出大量 NaN 行）
    * 默认起始 2016-01-01：沪深300ETF 成交量始于 2012-05-28，保证三个回归器
      从窗口首日起完整（fit 端 AutoTS 会把回归器 NaN 补 0，0 是假成交量，故
      必须预先保证完整）
    * 覆盖率过滤：窗口内非缺失占比 < min-coverage（默认 0.98）的股票剔除
      （晚上市/长期停牌会产生行首大片 NaN，AutoTS 的 na_tolerance 也会静默
      丢弃，这里显式剔除并在终端列出）
    * 回归器在 A 股交易日缺失（美股节假日、时差日）时 ffill→bfill 补齐
    * 数据量控制：AutoTS 训练前不会自动缩减数据（drop_data_older_than_periods
      默认关闭、na_tolerance 默认 1 不丢列，喂多少训多少），--tail N 只保留
      最近 N 个交易日；或在 AutoTS(drop_data_older_than_periods=N) 端截尾，
      二者取一即可

fit 时的预期行为（不是错误）：
    * df_cleanup（autots/tools/shaping.py）按 frequency='B' 给 df 补入 A 股
      节假日 NaN 行，随后打印 "future_regressor row count does not match"，并把
      回归器 reindex 到补齐后的索引、节假日行补 0 —— 与 fit 端对 NaN 回归器
      的默认处理一致，无需干预
    * 大量 Template Eval Error 是库内部分模板与新版本 scipy/numpy/pandas 的
      兼容问题，遗传搜索会自动跳过失败的模板继续搜索

训练示例（frequency 必须显式 'B'：infer 在春节/国庆长休下易失败）：

    from industry_dataset import load_industry_dataset
    from autots import AutoTS
    import pandas as pd

    df, regressor = load_industry_dataset()
    model = AutoTS(
        forecast_length=22,       # 未来 22 期；'B' 频率下预测索引会含节假日占位行
        frequency='B',
        ensemble='simple',
        model_list='superfast',
        max_generations=5,
        num_validations=2,
    )
    model.fit(df, future_regressor=regressor)

    # predict 需提供未来 forecast_length 期回归器；真实未来成交量未知，
    # 演示用末日值复制占位（更严谨可先对三条 volume 序列自建模预测）
    future_idx = pd.date_range(df.index[-1], periods=model.forecast_length + 1,
                               freq='B')[1:]
    future_reg = pd.concat([regressor.tail(1)] * model.forecast_length,
                           ignore_index=True)
    future_reg.index = future_idx
    prediction = model.predict(future_regressor=future_reg)
    print(prediction.forecast.head())

用法：
    python industry_dataset.py                          # 默认行业 白酒Ⅱ，打印概况并导出 CSV
    python industry_dataset.py --industry 半导体        # 换行业
    python industry_dataset.py --start 2015-01-01       # 换窗口起点
    python industry_dataset.py --tail 1250              # 只保留最近 1250 个交易日（约 5 年）
    python industry_dataset.py --min-coverage 0.95      # 放宽停牌容忍度
"""

import argparse
import os

import duckdb
import pandas as pd

DEFAULT_DB_PATH = "download/autots.duckdb"
DEFAULT_INDUSTRY = "电力"
DEFAULT_START = "2016-01-01"
DEFAULT_MIN_COVERAGE = 0.98
DEFAULT_OUT_DIR = "data"

# symbol -> 品种中文名（close 与 volume 都作为回归器）
MACRO_SYMBOLS = {
    "SSE:000001": "上证综指",
    "SSE:510300": "沪深300ETF",
    "SPCFD:SPX": "标普500",
}


def load_industry_dataset(industry=DEFAULT_INDUSTRY, start=DEFAULT_START,
                          min_coverage=DEFAULT_MIN_COVERAGE,
                          db_path=DEFAULT_DB_PATH, tail=None):
    """返回 (df_close, df_regressor, dropped_coverage)。

    df_close           DatetimeIndex 宽表，每列一只行业内股票的后复权收盘价（元），停牌日 NaN
    df_regressor       同 index 宽表，六列宏观回归器（各品种 close_/vol_ 各一列），无 NaN
    dropped_coverage   pd.Series，因覆盖率不足被剔除的股票列名 -> 覆盖率
    tail               只保留最近 N 个交易日（None = 全部窗口）
    """
    con = duckdb.connect(db_path, read_only=True)
    try:
        prices = con.execute(
            """
            SELECT a.symbol, a.date, a.close, s.name
            FROM akshare_tx a JOIN stock_list s ON a.symbol = s.symbol
            WHERE s.industry_l2 = ? AND a.adjust = 'hfq' AND a.date >= ?
            ORDER BY a.date
            """,
            [industry, start],
        ).fetchdf()
        macro = con.execute(
            """
            SELECT symbol, date, close, volume FROM tradingview_macro
            WHERE symbol IN ('SSE:000001', 'SSE:510300', 'SPCFD:SPX') AND date >= ?
            """,
            [start],
        ).fetchdf()
    finally:
        con.close()

    if prices.empty:
        raise ValueError(f"stock_list 中没有行业 {industry!r} 的 akshare_tx 数据")

    prices["date"] = pd.to_datetime(prices["date"])
    close_wide = prices.pivot(index="date", columns="symbol", values="close")
    sym_to_name = dict(zip(prices["symbol"], prices["name"]))
    close_wide.columns = [
        f"{sym}_{sym_to_name[sym]}" if pd.notna(sym_to_name.get(sym)) else sym
        for sym in close_wide.columns
    ]
    if tail is not None and len(close_wide) > tail:
        close_wide = close_wide.tail(tail)

    coverage = close_wide.notna().mean()
    keep_cols = coverage[coverage >= min_coverage].index.tolist()
    dropped = coverage[coverage < min_coverage].sort_values()
    close_wide = close_wide[keep_cols].sort_index()
    if close_wide.shape[1] == 0:
        raise ValueError(
            f"行业 {industry!r} 在 {start} 后无股票满足覆盖率 {min_coverage}，"
            f"可用 --start 后移或 --min-coverage 放宽"
        )

    macro["date"] = pd.to_datetime(macro["date"])
    macro_close = macro.pivot(index="date", columns="symbol", values="close")
    macro_vol = macro.pivot(index="date", columns="symbol", values="volume")
    macro_wide = pd.concat(
        [
            macro_close.rename(columns={s: f"close_{n}" for s, n in MACRO_SYMBOLS.items()}),
            macro_vol.rename(columns={s: f"vol_{n}" for s, n in MACRO_SYMBOLS.items()}),
        ],
        axis=1,
    )
    expected_cols = [f"{p}_{n}" for n in MACRO_SYMBOLS.values() for p in ("close", "vol")]
    missing = [c for c in expected_cols if c not in macro_wide.columns]
    if missing:
        raise ValueError(f"tradingview_macro 缺少品种数据，无法生成列：{missing}")
    macro_wide = macro_wide.reindex(close_wide.index).ffill().bfill()
    if macro_wide.isna().any().any():
        raise ValueError("回归器存在窗口内无法 ffill/bfill 补齐的缺失，请检查 --start 与品种数据")

    return close_wide, macro_wide[expected_cols], dropped


def main():
    parser = argparse.ArgumentParser(description="行业收盘价宽表 + 宏观成交量回归器")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="duckdb 路径")
    parser.add_argument("--industry", default=DEFAULT_INDUSTRY, help="industry_l2 行业名")
    parser.add_argument("--start", default=DEFAULT_START, help="窗口起始日（默认 2016-01-01）")
    parser.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE,
                        help="股票保留阈值：窗口内非缺失占比（默认 0.98）")
    parser.add_argument("--tail", type=int, default=None,
                        help="只保留最近 N 个交易日（默认全部；控制训练数据量）")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="CSV 输出目录")
    args = parser.parse_args()

    df, regressor, dropped = load_industry_dataset(
        industry=args.industry, start=args.start,
        min_coverage=args.min_coverage, db_path=args.db, tail=args.tail,
    )

    print(f"行业 {args.industry!r}  窗口 {df.index[0].date()} ~ {df.index[-1].date()}"
          f"  共 {len(df)} 个交易日")
    print(f"保留 {df.shape[1]} 只股票（--min-coverage={args.min_coverage}）")
    if len(dropped):
        print(f"剔除 {len(dropped)} 只（覆盖率不足，多为晚上市/长期停牌）：")
        for col, cov in dropped.items():
            print(f"    {col}  覆盖率 {cov:.1%}")
    nan_stat = df.isna().sum()
    if nan_stat.any():
        print("保留股票中的停牌 NaN 列（交给 AutoTS 模型层 fillna）：")
        for col, n in nan_stat[nan_stat > 0].items():
            print(f"    {col}  {n} 天")
    print(f"回归器列：{list(regressor.columns)}（无 NaN）")

    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.industry.replace("/", "_")
    close_path = os.path.join(args.out_dir, f"industry_{tag}_close.csv")
    reg_path = os.path.join(args.out_dir, f"industry_{tag}_regressor.csv")
    df.to_csv(close_path, encoding="utf-8-sig")
    regressor.to_csv(reg_path, encoding="utf-8-sig")
    print(f"已导出：\n    {close_path}\n    {reg_path}")


if __name__ == "__main__":
    main()
