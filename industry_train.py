# -*- coding: utf-8 -*-
"""行业股票收益率区间预测：AutoTS 训练 + 预测脚本（配合 industry_dataset.py）。

读取 industry_dataset.load_industry_dataset() 输出的行业对数收益率宽表 +
宏观回归器（上证综指/沪深300ETF/标普500 的 close/volume），用 AutoTS 遗传搜索
选模型，输出每只股票未来 --horizon 个（默认 1 = 明天）交易日的收益率预测区间。

输入（复用 industry_dataset.py，不重复读库）：
    df          DatetimeIndex 宽表，对数收益率 ln(P_t/P_{t-1})，停牌日 NaN
    regressor   同 index 六列宏观回归器（close_/vol_ × 3 品种）

输出（--out-dir，默认 data/）：
    industry_{行业}_forecast_{YYYYMMDD}.csv   预测区间明细（目标日命名）：
        预测日期          目标交易日（真实 A 股日历，非 'B' 推算）
        股票             列名，如 600519_贵州茅台
        下界log / 点log / 上界log     对数收益率区间（AutoTS 原始输出）
        下界% / 点预测% / 上界%       简单收益率 exp(x)-1，百分号数值
        区间宽度%        上界% - 下界%（区间不确定度）
        方向            偏多(下界>0) / 偏空(上界<0) / 区间含零
    industry_{行业}_template.json            模型模板（export_template，可 import_template 复用）

关键实现决策（为什么这么写）：
    * frequency 必须显式 'B'：infer 在春节/国庆长休下会失败（industry_dataset.py 已注明）
    * forecast_index 重贴为 holiday_calendar 的真实下一交易日（判定字段
      is_holiday_related：周末/法定假/调休补班日全覆盖，True = 不开市）：
      'B' 只认周六日，推算出的"明天"可能是 A 股节假日；模型本身只关心行序，
      日期标签换掉无副作用
    * 未来回归器两档：明日宏观 close/volume 在预测时点未知（上证/ETF 明日收盘、
      SPX 隔夜未走完），last=沿用末日值（随机游走近似，快），model=对六条回归器
      序列自建一个小 AutoTS 再预测（多期 horizon 时更合理）
    * metric_weighting 抬高 containment_weighting：该指标 = 验证集实际值落入
      上下界的比例，直接对齐"预测区间"目标，默认为 0 会让搜索只优化点预测
    * no_negatives 保持 False：收益率天然可负
    * horizon=1 时每折验证窗口只有 1 天，num_validations 给到 3+ 才有区分度
    * 训练/验证切分（validation_method='backwards'）：初始评估折 = 全量数据
      末尾 horizon 天做 holdout；之后每折再向前回溯 horizon 天，共
      num_validations+1 折（num_validations 不含首折）。遗传搜索各代只在
      首折打分，幸存模型（models_to_validate）跨全部折重评得总分；最终
      predict 前选中模型/集成用全量数据重新拟合。fit 结束会打印各折实际日期
    * --no-use-regressor 可整体关闭回归器（fit/predict 都不传）：模型清单里
      支持 regression_type='User' 的模型在无回归器时会跳过这些参数组合

CPU / GPU 机器：
    * AutoTS 绝大多数模型是 sklearn/statsmodels 系，吃 CPU 多核 → --n-jobs 调大
      （float=核数比例，如 0.9；int=精确进程数）
    * 只有 torch 系（PytorchForecasting/TiDE/NeuralForecast/MambaSSM/pMLP/TVAModel）
      会自动检测 CUDA（torch.cuda.is_available → lightning accelerator='gpu'）
    * GPU 机器建议 --model-list probabilistic（区间导向，含 ARCH/GARCH 族——
      对收益率波动率建模最对口）或 gpu（torch 系全集）；对应额外依赖：
          pip install arch                                  # ARCH 模型
          pip install torch lightning pytorch-forecasting   # torch 系模型
          pip install gluonts                               # GluonTS（可选）
          pip install neuralforecast                        # NeuralForecast（可选）
      装不全也没关系，AutoTS 对缺失依赖的模型自动跳过

用法：
    # 本机 CPU 冒烟（当前 .venv 无 torch/matplotlib，superfast 全可用）
    python industry_train.py --model-list superfast --generations 2 --num-validations 2 --tail 750

    # 常规训练（全量历史，约几分钟）
    python industry_train.py

    # GPU 机器全量（区间导向模型 + GARCH）
    python industry_train.py --model-list probabilistic --generations 15 --num-validations 5

    # 预测未来 5 个交易日（非"明天"场景）
    python industry_train.py --horizon 5 --future-regressor model

    # 不使用宏观回归器（纯时序，对照组/排查回归器贡献）
    python industry_train.py --no-use-regressor

模型模板复用（避免每日重复遗传搜索）：
    from autots import AutoTS
    model = AutoTS(..., initial_template='General')
    model.import_template('data/industry_电力_template.json', method='only_new_model')
    model.fit(df, future_regressor=regressor)   # 只重估模板里的模型参数
"""

import argparse
import os
import sys

import duckdb
import numpy as np
import pandas as pd

from industry_dataset import load_industry_dataset
from autots import AutoTS

DEFAULT_DB_PATH = "download/autots.duckdb"
DEFAULT_INDUSTRY = "电力"
DEFAULT_START = "2016-01-01"
DEFAULT_MIN_COVERAGE = 0.98
DEFAULT_OUT_DIR = "data"
DEFAULT_HORIZON = 1
DEFAULT_INTERVAL = 0.9
DEFAULT_MODEL_LIST = "superfast"
DEFAULT_GENERATIONS = 5
DEFAULT_NUM_VALIDATIONS = 3
DEFAULT_SEED = 2026

# 在默认权重基础上：压 smape、抬 containment（区间覆盖率）、压 runtime。
# containment = 验证集实际值落入 [lower, upper] 的比例，是"区间预测"的直接目标。
BASE_METRIC_WEIGHTING = {
    'smape_weighting': 5,
    'mae_weighting': 2,
    'rmse_weighting': 2,
    'made_weighting': 0.05,
    'mage_weighting': 0,
    'mle_weighting': 0,
    'imle_weighting': 0,
    'spl_weighting': 3,
    'containment_weighting': 0,
    'contour_weighting': 0.01,
    'runtime_weighting': 0.01,
    'oda_weighting': 0.001,
    'wasserstein_weighting': 0.01,
}
INTERVAL_METRIC_WEIGHTING = dict(
    BASE_METRIC_WEIGHTING,
    smape_weighting=2,
    containment_weighting=4,
    runtime_weighting=0.05,
)


def next_trading_days(last_date, n_days, db_path=DEFAULT_DB_PATH):
    """返回 last_date 之后 n_days 个 A 股交易日（Timestamp 列表）。

    holiday_calendar 语义（download/sources/holiday.py，timor.tech 源）：
        is_holiday_related = is_holiday OR is_workday_adjustment，
        即 周末 + 法定节假日 + 调休补班周六日 全部覆盖，True = 不开市。
        已用 akshare_tx 真实交易日期交叉验证 2013~2026 零冲突
        （仅 2020-01-31 疫情延长假、2024-02-09 除夕休市为 timor.tech 标准
        数据盲区，属历史日期不影响向前预测，交易所特例休市需人工留意）。
    日历覆盖不到（表只到 2026-12-31）时回退 pd.bdate_range 并告警。
    """
    con = duckdb.connect(db_path, read_only=True)
    try:
        cal = con.execute(
            "SELECT date, is_holiday_related FROM holiday_calendar WHERE date > ? "
            "ORDER BY date",
            [last_date],
        ).fetchdf()
    finally:
        con.close()

    cal["date"] = pd.to_datetime(cal["date"])
    got = [pd.Timestamp(d) for d in
           cal[~cal["is_holiday_related"].astype(bool)]["date"]]
    if len(got) >= n_days:
        return got[:n_days]

    # 日历覆盖不足（表只到 2026-12-31）：缺口按工作日历补，且剔除与已取日期的重复
    fallback = [d for d in pd.bdate_range(last_date, periods=n_days + 8)[1:]
                if d not in got][: n_days - len(got)]
    print(
        f"[警告] holiday_calendar 覆盖不足（需 {n_days} 个交易日，只取到 {len(got)} 个），"
        f"剩余按工作日历推算：{fallback}", file=sys.stderr
    )
    return got + fallback


def forecast_future_regressor(regressor, horizon, future_days, random_seed,
                              method="last"):
    """构造未来 horizon 期的回归器宽表（index=future_days）。

    last   沿用最后一个交易日的值（指数/成交量近似随机游走）
    model  对六条回归器序列各建一个小 AutoTS（superfast、无验证折）再预测
    """
    if method == "last":
        future_reg = pd.concat([regressor.tail(1)] * horizon, ignore_index=True)
    elif method == "model":
        reg_model = AutoTS(
            forecast_length=horizon,
            frequency='B',
            model_list='superfast',
            max_generations=2,
            num_validations=0,
            ensemble=None,
            random_seed=random_seed,
            verbose=0,
        )
        reg_model.fit(regressor)
        future_reg = reg_model.predict().forecast
        if future_reg.shape[0] < horizon:  # 模型极端失败时的兜底，正常不触发
            pad = pd.concat([regressor.tail(1)] * (horizon - len(future_reg)))
            future_reg = pd.concat([future_reg, pad])
    else:
        raise ValueError(f"未知 future-regressor 模式：{method}")
    future_reg.index = pd.DatetimeIndex(future_days)
    return future_reg


def print_validation_splits(model, horizon):
    """打印 AutoTS 实际使用的训练/验证切分（validation_method='backwards'）。

    validation_indexes 含 num_validations+1 个索引（第 0 个是初始评估折），
    每折按 simple_train_test_split 约定切：索引末尾 horizon 行为 holdout，
    其余为训练集。validate_num_validations 可能按数据长度自动缩减折数，
    故以 fit 后对象上的实际值为准。
    """
    print(f"\n===== 训练/验证切分（共 {len(model.validation_indexes)} 折，"
          f"每折末尾 {horizon} 天做 holdout）=====")
    for i, idx in enumerate(model.validation_indexes):
        if isinstance(idx, tuple):  # mixed_length 模式返回 (train, test) 元组
            train_idx, test_idx = idx
        else:
            train_idx, test_idx = idx[:-horizon], idx[-horizon:]
        test_desc = str(test_idx[0].date()) if horizon == 1 else (
            f"{test_idx[0].date()} ~ {test_idx[-1].date()}"
        )
        print(f"  fold {i}: 训练 {train_idx[0].date()} ~ {train_idx[-1].date()}"
              f"（{len(train_idx)} 天） → 预测 {test_desc}")
    print("  验证仅用于选模型/集成；最终 predict 前选中模型会用全量数据重新拟合")


def parse_n_jobs(raw):
    """AutoTS 的 n_jobs 接受 int（进程数）或 (0,1] float（核数比例）。"""
    value = float(raw)
    return int(value) if value >= 1 else value


def main():
    parser = argparse.ArgumentParser(description="行业股票收益率区间预测（AutoTS）")
    parser.add_argument("--industry", default=DEFAULT_INDUSTRY, help="industry_l2 行业名")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="duckdb 路径")
    parser.add_argument("--start", default=DEFAULT_START, help="窗口起始日")
    parser.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE)
    parser.add_argument("--tail", type=int, default=None,
                        help="只保留最近 N 个交易日（控制训练量/换手窗口）")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON,
                        help="预测期数（默认 1 = 明天）")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                        help="预测区间置信度（默认 0.9）")
    parser.add_argument("--model-list", default=DEFAULT_MODEL_LIST,
                        help="AutoTS 模型清单：superfast/fast/probabilistic/gpu/all 等")
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--num-validations", type=int, default=DEFAULT_NUM_VALIDATIONS,
                        help="回测折数（horizon=1 时建议 >=3）")
    parser.add_argument("--ensemble", default="simple",
                        help="集成方式：simple/dist/spatial/none")
    parser.add_argument("--future-regressor", choices=["last", "model"],
                        default="last", help="未来回归器构造方式")
    parser.add_argument("--use-regressor", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="是否使用宏观回归器（--no-use-regressor 时 fit/predict 全程不传）")
    parser.add_argument("--n-jobs", type=parse_n_jobs, default=0.5,
                        help="并行度：float(0~1]=核数比例，int>1=进程数")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--quiet", action="store_true", help="关掉 AutoTS 进度输出")
    args = parser.parse_args()

    ensemble = None if args.ensemble in ("none", "None") else args.ensemble

    df, regressor, dropped = load_industry_dataset(
        industry=args.industry, start=args.start,
        min_coverage=args.min_coverage, db_path=args.db, tail=args.tail,
        log_returns=True,
    )
    print(f"行业 {args.industry!r}  训练窗口 {df.index[0].date()} ~ {df.index[-1].date()}"
          f"  共 {len(df)} 个交易日 × {df.shape[1]} 只股票")
    if len(dropped):
        print(f"覆盖率不足剔除 {len(dropped)} 只：{', '.join(dropped.index[:8])}"
              f"{' ...' if len(dropped) > 8 else ''}")

    future_days = next_trading_days(df.index[-1], args.horizon, db_path=args.db)
    print(f"预测目标日：{'、'.join(str(d.date()) for d in future_days)}")

    model = AutoTS(
        forecast_length=args.horizon,
        frequency='B',
        prediction_interval=args.interval,
        model_list=args.model_list,
        max_generations=args.generations,
        num_validations=args.num_validations,
        validation_method='backwards',
        ensemble=ensemble,
        no_negatives=False,
        metric_weighting=INTERVAL_METRIC_WEIGHTING,
        random_seed=args.seed,
        n_jobs=args.n_jobs,
        verbose=0 if args.quiet else 1,
    )
    if args.use_regressor:
        model.fit(df, future_regressor=regressor)
    else:
        model.fit(df)
    print_validation_splits(model, args.horizon)

    print("\n===== 验证集模型排行（前 10）=====")
    leaderboard = model.results('validation')
    if isinstance(leaderboard, str):
        print(leaderboard)
    else:
        show_cols = ["Score"] + [
            c for c in ["smape", "containment", "mae", "runtime"]
            if c in leaderboard.columns
        ]
        print(leaderboard.head(10)[show_cols].to_string())

    if args.use_regressor:
        future_reg = forecast_future_regressor(
            regressor, args.horizon, future_days, args.seed,
            method=args.future_regressor,
        )
        print(f"\n未来回归器（{args.future_regressor} 模式）首日取值：")
        print(future_reg.head(1).round(2).to_string())
        prediction = model.predict(
            forecast_length=args.horizon,
            prediction_interval=args.interval,
            future_regressor=future_reg,
        )
    else:
        prediction = model.predict(
            forecast_length=args.horizon,
            prediction_interval=args.interval,
        )

    # 'B' 推算的未来索引换成真实 A 股交易日标签
    point = prediction.forecast.copy()
    lower = prediction.lower_forecast.copy()
    upper = prediction.upper_forecast.copy()
    for frame in (point, lower, upper):
        frame.index = pd.DatetimeIndex(future_days)

    tidy = pd.DataFrame(
        {
            "下界log": lower.stack(),
            "点log": point.stack(),
            "上界log": upper.stack(),
        }
    ).rename_axis(["预测日期", "股票"]).reset_index()
    for col in ["下界", "点", "上界"]:
        tidy[f"{col}预测%" if col == "点" else f"{col}%"] = (
            np.expm1(tidy[f"{col}log"]) * 100
        )
    tidy["区间宽度%"] = tidy["上界%"] - tidy["下界%"]
    tidy["方向"] = np.where(
        tidy["下界%"] > 0, "偏多",
        np.where(tidy["上界%"] < 0, "偏空", "区间含零"),
    )
    tidy = tidy.sort_values(["预测日期", "点预测%"], ascending=[True, False])

    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.industry.replace("/", "_")
    forecast_path = os.path.join(
        args.out_dir, f"industry_{tag}_forecast_{future_days[0]:%Y%m%d}.csv"
    )
    template_path = os.path.join(args.out_dir, f"industry_{tag}_template.json")
    tidy.to_csv(forecast_path, index=False, encoding="utf-8-sig")
    model.export_template(template_path, models='best', n=10)

    tomorrow = tidy[tidy["预测日期"] == future_days[0]]
    print(f"\n===== 明日（{future_days[0].date()}）区间预测概览 =====")
    print(f"偏多 {int((tomorrow['方向'] == '偏多').sum())} 只 | "
          f"偏空 {int((tomorrow['方向'] == '偏空').sum())} 只 | "
          f"区间含零 {int((tomorrow['方向'] == '区间含零').sum())} 只")
    print(f"区间宽度中位数 {tomorrow['区间宽度%'].median():.2f}%   "
          f"点预测绝对值中位数 {tomorrow['点预测%'].abs().median():.2f}%")
    show = ["股票", "下界%", "点预测%", "上界%", "区间宽度%", "方向"]
    print("\n点预测最高 5 只：")
    print(tomorrow.head(5)[show].to_string(index=False))
    print("\n点预测最低 5 只：")
    print(tomorrow.tail(5)[show].to_string(index=False))

    print(f"\n已导出：\n    {forecast_path}\n    {template_path}")
    if isinstance(leaderboard, pd.DataFrame):
        print(f"获胜模型：{model.best_model['Model'].iloc[0]}")


if __name__ == "__main__":
    main()
