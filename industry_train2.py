# -*- coding: utf-8 -*-
"""行业股票收益率区间预测：30 天滚动回测（industry_train.py 的验证版副本）。

协议（严格防数据泄露）：
    1. 全量数据（对数收益率宽表 + 宏观回归器）的末尾 --test-days（默认 30）
       个交易日整体扣下作为测试集，任何训练/模型选择都不得触碰；
    2. 用前段数据跑一次完整遗传搜索（选模型 + 集成），导出模板
       （unpack_ensembles=True：拆成纯组件模型，避免集成行引用的子模型
       不在模板内导致导入失败）；
    3. 滚动预测 i = 0..test_days-1：训练窗口 = 前段数据 + 测试集前 i 天
       （预测第 i+1 天时，第 i 天及之前全是"已知历史"，合法），
       用导入模板 + max_generations=0 + num_validations=0 重新拟合
       （generation 0 无条件评估导入模板 → 单折验证选出当期最优组件），
       预测下一交易日的收益率区间；
    4. 未来回归器用 last 模式（沿用窗口末日值）：目标日的宏观 close/volume
       在预测时点未知，回测中虽是历史数据但取用即泄露，与实盘口径保持一致。

信号与命中定义（下界>0 且 区间中值>阈值 → 记作正收益，阈值 --mid-threshold）：
    区间中值% = (下界% + 上界%) / 2 的简单收益率
    信号   当日该股 下界% > 0（区间整体为正，强过滤）且 区间中值% > --mid-threshold
    命中   信号成立 且 当日实际简单收益率 > 0
    命中率 = 命中次数 / 信号次数（每只股票最多 test-days 次预测）

输出（--out-dir，默认 data/）：
    industry_{行业}_backtest_{起}_{止}.csv        逐日逐股明细：
        预测日期 / 股票 / 下界% / 点预测% / 上界% / 区间宽度% /
        预测方向 / 实际% / 实际上涨 / 信号 / 命中
    industry_{行业}_backtest_summary_{止}.csv    逐股汇总：
        信号次数 / 命中次数 / 命中率 / 实际上涨天数 / 实际上涨占比
    industry_{行业}_template2.json               搜索阶段导出的模型模板

用法：
    # 本机 CPU 冒烟（小窗口、少折数）
    python industry_train2.py --generations 2 --num-validations 2 --tail 500 --test-days 5

    # 完整 30 天回测（约 = 一次常规训练 + 30 次模板重拟合）
    python industry_train2.py

    # GPU 机器
    python industry_train2.py --model-list probabilistic --generations 15 --num-validations 5
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from industry_dataset import load_industry_dataset
from autots import AutoTS

DEFAULT_DB_PATH = "download/autots.duckdb"
DEFAULT_INDUSTRY = "电力"
DEFAULT_START = "2016-01-01"
DEFAULT_MIN_COVERAGE = 0.98
DEFAULT_OUT_DIR = "data"
DEFAULT_TEST_DAYS = 30
DEFAULT_INTERVAL = 0.9
DEFAULT_MODEL_LIST = "superfast"
DEFAULT_GENERATIONS = 5
DEFAULT_NUM_VALIDATIONS = 3
DEFAULT_TEMPLATE_N = 15
DEFAULT_SEED = 2026

# 与 industry_train.py 一致：压 smape、抬 containment（区间覆盖率）、压 runtime
INTERVAL_METRIC_WEIGHTING = {
    'smape_weighting': 2,
    'mae_weighting': 2,
    'rmse_weighting': 2,
    'made_weighting': 0.05,
    'mage_weighting': 0,
    'mle_weighting': 0,
    'imle_weighting': 0,
    'spl_weighting': 3,
    'containment_weighting': 4,
    'contour_weighting': 0.01,
    'runtime_weighting': 0.05,
    'oda_weighting': 0.001,
    'wasserstein_weighting': 0.01,
}


def parse_n_jobs(raw):
    """AutoTS 的 n_jobs 接受 int（进程数）或 (0,1] float（核数比例）。"""
    value = float(raw)
    return int(value) if value >= 1 else value


def build_model(args, forecast_length, max_generations, num_validations, ensemble,
                template=None):
    """构造 AutoTS 对象；template 非 None 时直接作为初始模板（import 'only'）。"""
    kwargs = dict(
        forecast_length=forecast_length,
        frequency='B',
        prediction_interval=args.interval,
        model_list=args.model_list,
        max_generations=max_generations,
        num_validations=num_validations,
        validation_method='backwards',
        ensemble=ensemble,
        no_negatives=False,
        metric_weighting=INTERVAL_METRIC_WEIGHTING,
        random_seed=args.seed,
        n_jobs=args.n_jobs,
        verbose=0,
    )
    if template is not None:
        # initial_template 传 DataFrame = __init__ 内直接 import_template(method='only')
        kwargs["initial_template"] = template.copy()
    return AutoTS(**kwargs)


def run_search(df_train, reg_train, args):
    """搜索阶段：仅用训练段数据做遗传搜索，返回 (model, template_df)。"""
    model = build_model(args, forecast_length=1,
                        max_generations=args.generations,
                        num_validations=args.num_validations,
                        ensemble=args.ensemble)
    if reg_train is not None:
        model.fit(df_train, future_regressor=reg_train)
    else:
        model.fit(df_train)

    # 拆出纯组件模型：集成行的 ID 引用子模型，子模型不在导出集内会导致
    # 滚动重拟合导入失败；拆开后每步从组件中按单折验证重新选优
    template = model.export_template(
        None, models='best', n=args.template_n,
        unpack_ensembles=True, include_ensemble=False,
    )
    return model, template


def refit_and_predict(df_hist, reg_hist, template, args):
    """滚动单步：用历史窗口重拟合模板并预测下一交易日，返回 (lower, point, upper)。"""
    model = build_model(args, forecast_length=1,
                        max_generations=0, num_validations=0, ensemble=None,
                        template=template)
    if reg_hist is not None:
        model.fit(df_hist, future_regressor=reg_hist)
        future_reg = reg_hist.tail(1).copy()
        prediction = model.predict(forecast_length=1,
                                   prediction_interval=args.interval,
                                   future_regressor=future_reg)
    else:
        model.fit(df_hist)
        prediction = model.predict(forecast_length=1,
                                   prediction_interval=args.interval)
    return (prediction.lower_forecast.iloc[0].copy(),
            prediction.forecast.iloc[0].copy(),
            prediction.upper_forecast.iloc[0].copy())


def main():
    parser = argparse.ArgumentParser(
        description="行业股票收益率区间预测：30 天滚动回测（防泄露验证）")
    parser.add_argument("--industry", default=DEFAULT_INDUSTRY, help="industry_l2 行业名")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="duckdb 路径")
    parser.add_argument("--start", default=DEFAULT_START, help="窗口起始日")
    parser.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE)
    parser.add_argument("--tail", type=int, default=None,
                        help="全量窗口只保留最近 N 个交易日（先截尾再切测试集）")
    parser.add_argument("--test-days", type=int, default=DEFAULT_TEST_DAYS,
                        help="末尾保留的回测交易日数（默认 30）")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                        help="预测区间置信度（默认 0.9）")
    parser.add_argument("--mid-threshold", type=float, default=1.0,
                        help="信号的区间中值%%阈值（信号 = 下界>0 且 中值>该值，默认 1.0）")
    parser.add_argument("--model-list", default=DEFAULT_MODEL_LIST,
                        help="AutoTS 模型清单：superfast/fast/probabilistic/gpu/all 等")
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS,
                        help="搜索阶段遗传代数（滚动阶段固定 0 代，只重估模板）")
    parser.add_argument("--num-validations", type=int, default=DEFAULT_NUM_VALIDATIONS,
                        help="搜索阶段回测折数")
    parser.add_argument("--ensemble", default="simple",
                        help="搜索阶段集成方式：simple/dist/spatial/none")
    parser.add_argument("--template-n", type=int, default=DEFAULT_TEMPLATE_N,
                        help="导出进滚动模板的模型个数")
    parser.add_argument("--use-regressor", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="是否使用宏观回归器（--no-use-regressor 时 fit/predict 全程不传）")
    parser.add_argument("--n-jobs", type=parse_n_jobs, default=0.5,
                        help="并行度：float(0~1]=核数比例，int>1=进程数")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--quiet", action="store_true", help="关掉搜索阶段 AutoTS 进度输出")
    args = parser.parse_args()

    ensemble = None if args.ensemble in ("none", "None") else args.ensemble

    df, regressor, dropped = load_industry_dataset(
        industry=args.industry, start=args.start,
        min_coverage=args.min_coverage, db_path=args.db, tail=args.tail,
        log_returns=True,
    )
    if len(df) <= args.test_days + 60:
        raise ValueError(
            f"窗口只剩 {len(df)} 个交易日，扣除 {args.test_days} 天测试集后训练数据过短，"
            f"请减小 --tail / --test-days 或前移 --start"
        )

    test = df.iloc[-args.test_days:]
    train = df.iloc[:-args.test_days]
    reg_train = regressor.loc[train.index] if args.use_regressor else None
    print(f"行业 {args.industry!r}  共 {len(df)} 个交易日 × {df.shape[1]} 只股票")
    print(f"训练段 {train.index[0].date()} ~ {train.index[-1].date()}（{len(train)} 天）  "
          f"测试段 {test.index[0].date()} ~ {test.index[-1].date()}（{len(test)} 天）")
    if len(dropped):
        print(f"覆盖率不足剔除 {len(dropped)} 只：{', '.join(dropped.index[:8])}"
              f"{' ...' if len(dropped) > 8 else ''}")

    print("\n===== 阶段 1/2：搜索段遗传搜索（仅训练段数据）=====")
    t0 = time.time()
    model, template = run_search(train, reg_train, args)
    print(f"搜索完成，用时 {time.time() - t0:.0f}s，导出 {len(template)} 个组件模型进滚动模板：")
    print(template[['Model']].value_counts().to_string())

    leaderboard = model.results('validation')
    if isinstance(leaderboard, pd.DataFrame):
        show_cols = ["Score"] + [c for c in ["smape", "containment", "mae", "runtime"]
                                 if c in leaderboard.columns]
        print("\n验证集模型排行（前 10）：")
        print(leaderboard.head(10)[show_cols].to_string())

    print("\n===== 阶段 2/2：滚动重拟合 + 单日预测 =====")
    rows = []
    n_train = len(train)
    t0 = time.time()
    for i, target_date in enumerate(test.index):
        df_hist = df.iloc[:n_train + i]
        reg_hist = regressor.loc[df_hist.index] if args.use_regressor else None
        try:
            lower, point, upper = refit_and_predict(df_hist, reg_hist, template, args)
        except Exception as exc:  # 单步失败不终止回测：记 NaN 继续后续日期
            print(f"[警告] {target_date.date()} 重拟合失败：{exc}", file=sys.stderr)
            lower = point = upper = pd.Series(np.nan, index=df.columns)
        for stock in df.columns:
            rows.append({
                "预测日期": target_date,
                "股票": stock,
                "下界%": np.expm1(lower.get(stock, np.nan)) * 100,
                "点预测%": np.expm1(point.get(stock, np.nan)) * 100,
                "上界%": np.expm1(upper.get(stock, np.nan)) * 100,
                "实际%": np.expm1(df.at[target_date, stock]) * 100
                if pd.notna(df.at[target_date, stock]) else np.nan,
            })
        elapsed = time.time() - t0
        print(f"  [{i + 1}/{len(test)}] 预测 {target_date.date()} 完成"
              f"（累计 {elapsed:.0f}s，预计剩余 {(elapsed / (i + 1)) * (len(test) - i - 1):.0f}s）")

    detail = pd.DataFrame(rows)
    detail["区间宽度%"] = detail["上界%"] - detail["下界%"]
    detail["区间中值%"] = (detail["下界%"] + detail["上界%"]) / 2
    detail["预测方向"] = np.where(
        detail["区间中值%"] > 0, "偏多",
        np.where(detail["区间中值%"] < 0, "偏空", "持平"),
    )
    detail["实际上涨"] = detail["实际%"] > 0
    detail["信号"] = (detail["下界%"] > 0) & (detail["区间中值%"] > args.mid_threshold)
    detail["命中"] = detail["信号"] & detail["实际上涨"]
    detail = detail.sort_values(["预测日期", "点预测%"], ascending=[True, False])

    # 逐股汇总：停牌日（实际% 为 NaN）不计入命中统计
    valid = detail[detail["实际%"].notna()]
    summary = valid.groupby("股票").agg(
        信号次数=("信号", "sum"),
        命中次数=("命中", "sum"),
        实际上涨天数=("实际上涨", "sum"),
        预测天数=("实际%", "count"),
    ).reset_index()
    summary["命中率%"] = np.where(
        summary["信号次数"] > 0, summary["命中次数"] / summary["信号次数"] * 100, np.nan)
    summary["实际上涨占比%"] = summary["实际上涨天数"] / summary["预测天数"] * 100
    summary = summary.sort_values("命中率%", ascending=False, na_position="last")

    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.industry.replace("/", "_")
    detail_path = os.path.join(
        args.out_dir,
        f"industry_{tag}_backtest_{test.index[0]:%Y%m%d}_{test.index[-1]:%Y%m%d}.csv",
    )
    summary_path = os.path.join(
        args.out_dir, f"industry_{tag}_backtest_summary_{test.index[-1]:%Y%m%d}.csv")
    template_path = os.path.join(args.out_dir, f"industry_{tag}_template2.json")
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    model.export_template(template_path, models='best', n=args.template_n)

    n_signal = int(valid["信号"].sum())
    n_hit = int(valid["命中"].sum())
    base_rate = valid["实际上涨"].mean() * 100
    print(f"\n===== 回测结果（{test.index[0].date()} ~ {test.index[-1].date()}，"
          f"{len(test)} 天 × {df.shape[1]} 只）=====")
    if n_signal > 0:
        print(f"发出信号（下界% > 0 且 区间中值% > {args.mid_threshold}）{n_signal} 次，"
              f"命中 {n_hit} 次，命中率 {n_hit / n_signal * 100:.1f}%")
    else:
        print("发出信号 0 次")
    print(f"对照：全部股票-日实际上涨占比 {base_rate:.1f}%（信号命中率高于它才有增量价值）")
    by_day = valid.groupby("预测日期").agg(
        信号数=("信号", "sum"), 命中数=("命中", "sum"))
    by_day["当日命中率%"] = np.where(
        by_day["信号数"] > 0, by_day["命中数"] / by_day["信号数"] * 100, np.nan)
    print("\n逐日信号量与命中率：")
    print(by_day.to_string(float_format=lambda x: f"{x:.0f}"))

    print("\n逐股命中率（信号次数 ≥ 3 的前 10 / 后 10）：")
    ranked = summary[summary["信号次数"] >= 3]
    if len(ranked):
        print(ranked.head(10).to_string(index=False, float_format=lambda x: f"{x:.1f}"))
        print("...")
        print(ranked.tail(10).to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    else:
        print("（没有股票发出 ≥3 次信号）")

    print(f"\n已导出：\n    {detail_path}\n    {summary_path}\n    {template_path}")


if __name__ == "__main__":
    main()
