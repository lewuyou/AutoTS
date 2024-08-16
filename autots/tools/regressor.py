import numpy as np
import pandas as pd
from autots.tools.impute import FillNA
from autots.tools.shaping import infer_frequency
from autots.tools.seasonal import date_part
from autots.tools.holiday import holiday_flag
from autots.tools.cointegration import coint_johansen
from autots.evaluator.anomaly_detector import HolidayDetector
from autots.tools.transform import GeneralTransformer


def create_regressor(
    df,
    forecast_length,
    frequency: str = "infer",
    holiday_countries: list = ["US"],
    datepart_method: str = "simple_binarized",
    drop_most_recent: int = 0,
    scale: bool = True,
    summarize: str = "auto",
    backfill: str = "bfill",
    n_jobs: str = "auto",
    fill_na: str = 'ffill',
    aggfunc: str = "first",
    encode_holiday_type=False,
    holiday_detector_params={
        "threshold": 0.8,
        "splash_threshold": None, # 设定一个界限，用于区分显著的假日效应
        "use_dayofmonth_holidays": True, # 月份中固定日期的假日
        "use_wkdom_holidays": True, # 月初的工作日假期
        "use_wkdeom_holidays": False, # 月底的工作日假期
        "use_lunar_holidays": True, # 农历假期，如春节
        "use_lunar_weekday": False, # 农历的工作日
        "use_islamic_holidays": False, # 伊斯兰假期
        "use_hebrew_holidays": False, # 希伯来假期（例如，犹太新年）
        "output": 'univariate',
        "anomaly_detector_params": { # 异常检测器参数
            "method": "mad", # 异常检测方法，例如"mad"代表中位数绝对偏差
            "transform_dict": {
                "fillna": None,
                "transformations": {"0": "DifferencedTransformer"},
                "transformation_params": {"0": {}},
            },
            "forecast_params": None,
            "method_params": {"distribution": "gamma", "alpha": 0.05},
        },
    },
    holiday_regr_style: str = "flag",
    preprocessing_params: dict = None,
):
    """从现有数据集中可用信息创建回归器。
    组成部分：滞后数据、日期部分信息和假期。

    这个函数让人困惑。这不是机器学习模型所必需的，在 AutoTS 中，它们内部会单独创建更复杂的特征集。
    相反，这可能会帮助一些其他模型（GLM，ARIMA），这些模型接受回归量但不会内部构建回归特征集。
    并且这允许在输入到 AutoTS 之前根据需要进行事后自定义。

    推荐丢弃 regressor_train 和训练用的 df 的 .head(forecast_length)。
    `df = df.iloc[forecast_length:]`
    如果你不想要滞后特征，设置 summarize="median"，这将只给出一个此类列，然后可以轻松删除

    参数:
        df (pd.DataFrame): WIDE 格式的数据框（如果数据不是已经这样的话，使用 long_to_wide 转换）
            如果存在，将丢弃分类系列
        forecast_length (int): 将要预测的时间长度
        frequency (str): 你总是不得不使用的那些烦人的时间序列偏移代码
        holiday_countries (list): 列出要获取假期的国家。需要 holidays 包
            也可以是一个包含子区域（州）的字典 {'country': "subdiv"}
        datepart_method (str): 参见 seasonal 的 date_part
        scale (bool): 如果为真，使用 StandardScaler 标准化特征
        summarize (str): 如果特征很大，选择特征汇总方式:
            'pca', 'median', 'mean', 'mean+std', 'feature_agglomeration', 'gaussian_random_projection'
        backfill (str): 处理通过位移创建的 NaN 的方法
            "bfill"- 用最后的值向后填充
            "ETS" - 用 ETS 向后预测来填充
            "DatepartRegression" - 用 DatepartRegression 填充
        fill_na (str): 预填充数据中 NAs 的方法，可用的方法与其他地方相同
        aggfunc (str): 如果频率被重新采样，则使用的 str 或 func
        encode_holiday_type (bool): 如果为真，为每个假期返回一列，仅适用于 holidays 包的国家假期（不适用于 Detector）
        holiday_detector_params (dict): 传递给 HolidayDetector，或为 None
        holiday_regr_style (str): 传递给探测器的 dates_to_holidays 'flag', 'series_flag', 'impact'
        preprocessing_params (dict): 在创建回归器之前应用的 GeneralTransformer 参数

    返回值:
        regressor_train, regressor_forecast
    """
    if not isinstance(df.index, pd.DatetimeIndex): # 确保数据是 'wide' 格式
        raise ValueError(
            "create_regressor input df must be `wide` style with pd.DatetimeIndex index"
        )
    if isinstance(df, pd.Series): # 确保数据是 'wide' 格式
        df = df.to_frame()
    if drop_most_recent > 0: # 丢弃最近指定数量的数据
        df = df.drop(df.tail(drop_most_recent).index)
    if frequency == "infer": # 从数据中推断频率
        frequency = infer_frequency(df)
    else:
        # 用 NaN 填充索引中缺失的日期，根据需要重新采样为 freq
        try:
            if aggfunc == "first":
                df = df.resample(frequency).first()
            else:
                df = df.resample(frequency).apply(aggfunc)
        except Exception:
            df = df.asfreq(frequency, fill_value=None)
    # handle categorical
    # df = df.apply(pd.to_numeric, errors='ignore') # another pointless pandas deprecation
    for col in df.columns: #尝试将所有列转换为数值类型
        try:
            df.loc[:, col] = pd.to_numeric(df[col])
        except ValueError:
            pass
    df = df.select_dtypes(include=np.number) # 选择数值类型的列
    # macro_micro
    if preprocessing_params is not None: # 在创建回归器之前应用的 GeneralTransformer 参数
        trans = GeneralTransformer(**preprocessing_params)
        df = trans.fit_transform(df) # 用于数据的标准化、归一化或其他形式的预处理

    # 数据滞后
    '''
    5.标准化数据
    6.将数据降维到10维
    7.填充缺失日期（已填充）对应的数据
    8.将假期编码为二进制
    9.返回结尾部分forecast_length(预测长度)的数据
    10.预测部分数据重置索引，时间设置为最后日期往后
    11.regressor_train 数据时间索引整体推迟60天(forecast_length)
    regressor_train + regressor_forecast 的数据等于原来的数据日期整体往后移动60天
    12.对因为数据移位产生的 NaN 进行填充,使用bfill方法,如果使用ets或者datepartregression,
    ,则调用model_forecast对空缺数据进行预测。
    '''
    regr_train, regr_fcst = create_lagged_regressor(
        df,
        forecast_length=forecast_length,
        frequency=frequency,
        summarize=summarize,
        scale=scale,  # already done above
        backfill=backfill,
        fill_na=fill_na,
    )
    # datepart,增加几月份、几号、周末、时间戳、周几的标记
    if datepart_method is not None:
        regr_train = pd.concat(
            [regr_train, date_part(regr_train.index, method=datepart_method)],
            axis=1,
        )
        regr_fcst = pd.concat(
            [regr_fcst, date_part(regr_fcst.index, method=datepart_method)],
            axis=1,
        )
    # holiday (list)
    if holiday_countries is not None:
        if isinstance(holiday_countries, str):
            holiday_countries = holiday_countries.split(",")
        if isinstance(holiday_countries, list):
            holiday_countries = {x: None for x in holiday_countries}

        for holiday_country, holidays_subdiv in holiday_countries.items():
            # create holiday flag for historic regressor
            regr_train = pd.concat(
                [
                    regr_train,
                    holiday_flag(
                        regr_train.index,
                        country=holiday_country,
                        holidays_subdiv=holidays_subdiv,
                        encode_holiday_type=encode_holiday_type,
                    ),
                ],
                axis=1,
            )
            # now do again for future regressor
            regr_fcst = pd.concat(
                [
                    regr_fcst,
                    holiday_flag(
                        regr_fcst.index,
                        country=holiday_country,
                        holidays_subdiv=holidays_subdiv,
                        encode_holiday_type=encode_holiday_type,
                    ),
                ],
                axis=1,
            )
            # now try it for future days
            try:
                holiday_future = holiday_flag(
                    regr_train.index.shift(1, freq=frequency),
                    country=holiday_country,
                    holidays_subdiv=holidays_subdiv,
                )
                holiday_future.index = regr_train.index
                holiday_future_2 = holiday_flag(
                    regr_fcst.index.shift(1, freq=frequency),
                    country=holiday_country,
                    holidays_subdiv=holidays_subdiv,
                )
                holiday_future_2.index = regr_fcst.index
                regr_train[f"holiday_flag_{holiday_country}_future"] = holiday_future
                regr_fcst[f"holiday_flag_{holiday_country}_future"] = holiday_future_2
            except Exception:
                print(
                    f"holiday_future columns failed to add for {holiday_country}, likely due to complex datetime index"
                )
    if holiday_detector_params is not None:
        try:
            mod = HolidayDetector(**holiday_detector_params)
            mod.detect(df)
            train_holidays = mod.dates_to_holidays(
                regr_train.index, style=holiday_regr_style
            )
            fcst_holidays = mod.dates_to_holidays(
                regr_fcst.index, style=holiday_regr_style
            )
            all_cols = train_holidays.columns.union(fcst_holidays.columns)
            regr_train = pd.concat(
                [regr_train, train_holidays.reindex(columns=all_cols).fillna(0)],
                axis=1,
            )
            regr_fcst = pd.concat(
                [regr_fcst, fcst_holidays.reindex(columns=all_cols).fillna(0)],
                axis=1,
            )
        except Exception as e:
            print("HolidayDetector failed with error: " + repr(e)[:180])

    # columns all as strings
    regr_train.columns = [str(xc) for xc in regr_train.columns]
    regr_fcst.columns = [str(xc) for xc in regr_fcst.columns]
    # drop duplicate columns (which come from holidays)
    regr_train = regr_train.loc[:, ~regr_train.columns.duplicated()]
    regr_fcst = regr_fcst.loc[:, ~regr_fcst.columns.duplicated()]
    regr_fcst = regr_fcst.reindex(columns=regr_train.columns, fill_value=0)
    return regr_train, regr_fcst


def create_lagged_regressor(
    df,
    forecast_length: int,
    frequency: str = "infer",
    scale: bool = True,
    summarize: str = None,
    backfill: str = "bfill",
    n_jobs: str = "auto",
    fill_na: str = 'ffill',
):
    """创建特征的回归器，特征通过预测长度进行滞后。
    对于一些不使用此类信息的模型来说很有用。

    推荐丢弃 regressor_train 和训练用的 df 的 .head(forecast_length)。
    `df = df.iloc[forecast_length:]`

    参数:
        df (pd.DataFrame): 训练数据
        forecast_length (int): 预测长度，用于数据位移
        frequency (str): 对于日期时间相关操作非常必要的频率，默认为 'infer'
        scale (bool): 如果为 True，则使用 StandardScaler 标准化特征
        summarize (str): 如果特征很大，选择特征汇总方式:
            'pca', 'median', 'mean', 'mean+std', 'feature_agglomeration', 'gaussian_random_projection', "auto"
        backfill (str): 处理通过位移创建的 NaN 的方法
            "bfill"- 用最后的值向后填充
            "ETS" - 用 ETS 向后预测来填充
            "DatepartRegression" - 用 DatepartRegression 填充
        fill_na (str): 预填充数据中 NAs 的方法，可用的方法与其他地方相同

    返回值:
        regressor_train, regressor_forecast
    """
    model_flag = False
    if frequency == "infer": # 从数据中推断频率
        frequency = infer_frequency(df)
    if not isinstance(df.index, pd.DatetimeIndex): # 确保数据是 'wide' 格式
        raise ValueError("df must be a 'wide' dataframe with a pd.DatetimeIndex.")
    if isinstance(summarize, str): # 将 summarize 转换为小写
        summarize = summarize.lower()
    if isinstance(backfill, str):
        backfill = backfill.lower()
    if isinstance(df, pd.Series):
        df = df.to_frame()
    dates = df.index # 获取索引
    df_cols = df.columns
    df_inner = df.copy()

    if scale: # 如果为真，使用 StandardScaler 标准化特征
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        df_inner = pd.DataFrame(
            scaler.fit_transform(df_inner), index=dates, columns=df_cols
        )

    ag_flag = False
    # 下面的汇总方式不关心是否存在NaN
    if summarize is None:
        pass
    if summarize == "auto": # 自动选择特征维度汇总方式
        ag_flag = True if df_inner.shape[1] > 10 else False
    elif summarize == 'mean': # 求均值
        df_inner = df_inner.mean(axis=1).to_frame()
    elif summarize == 'median': # 求中位数
        df_inner = df_inner.median(axis=1).to_frame()
    elif summarize == 'mean+std': # 求均值和标准差
        df_inner = pd.concat(
            [df_inner.mean(axis=1).to_frame(), df_inner.std(axis=1).to_frame()], axis=1
        )
        df_inner.columns = [0, 1]
        
    # 填充假期的空数据（前面已新建假期索引日期），然后数据降维方法选择
    df_inner = FillNA(df_inner, method=fill_na)
    # some debate over whether PCA or RandomProjection will result in minor data leakage, if used
    if summarize == 'pca': # 主成分分析降维
        from sklearn.decomposition import PCA

        n_components = "mle" if df_inner.shape[0] > df_inner.shape[1] else None
        df_inner = FillNA(df_inner, method=fill_na)
        df_inner = pd.DataFrame(
            PCA(n_components=n_components).fit_transform(df_inner), index=dates
        )
        ag_flag = True if df_inner.shape[1] > 10 else False
    elif summarize == 'cointegration': # 协整分析降维
        ev, components_ = coint_johansen(df_inner.values, 0, 1, return_eigenvalues=True)
        df_inner = pd.DataFrame(
            np.matmul(components_, (df_inner.values).T).T,
            index=df_inner.index,
        ).iloc[:, np.flipud(np.argsort(ev))[0:10]]
    elif summarize == "feature_agglomeration" or ag_flag: # 特征聚合降维
        from sklearn.cluster import FeatureAgglomeration

        n_clusters = 10 if ag_flag else 25 # 聚类数,如果summarize=auto聚类数为10，否则为25
        if df_inner.shape[1] > 25:
            df_inner = pd.DataFrame(
                FeatureAgglomeration(n_clusters=n_clusters).fit_transform(df_inner),
                index=dates,
            )
    elif summarize == "gaussian_random_projection": # 高斯随机投影
        from sklearn.random_projection import GaussianRandomProjection

        df_inner = pd.DataFrame(
            GaussianRandomProjection(n_components='auto', eps=0.2).fit_transform(
                df_inner
            ),
            index=dates,
        )

    regressor_forecast = df_inner.tail(forecast_length)# 返回最后n行数据
    try:
        # 重置索引，将开始日期设置为数据最后一天，依次往后延申
        regressor_forecast.index = pd.date_range(
            dates[-1], periods=(forecast_length + 1), freq=frequency
        )[1:]
    except Exception:
        raise ValueError(
            "create_regressor doesn't work on data where forecast_length > historical data length"
        )
    # 数据时间索引整体推迟60天
    regressor_train = df_inner.shift(forecast_length)
    # 处理因为数据移位产生的 NaN 进行填充的方法
    if backfill == "ets":
        model_flag = True
        model_name = "ETS"
        model_param_dict = '{"damped_trend": false, "trend": "additive", "seasonal": null, "seasonal_periods": null}'
    elif backfill == 'datepartregression':
        model_flag = True
        model_name = 'DatepartRegression'
        model_param_dict = '{"regression_model": {"model": "RandomForest", "model_params": {}}, "datepart_method": "recurring", "regression_type": null}'
    else:
        regressor_train = regressor_train.bfill().ffill()

    if model_flag:
        from autots import model_forecast

        df_train = df_inner.iloc[::-1]
        df_train.index = dates
        df_forecast = model_forecast(
            model_name=model_name,
            model_param_dict=model_param_dict,
            model_transform_dict={
                'fillna': 'fake_date',
                'transformations': {'0': 'ClipOutliers'},
                'transformation_params': {'0': {'method': 'clip', 'std_threshold': 3}},
            },
            df_train=df_train,
            forecast_length=forecast_length,
            frequency=frequency,
            random_seed=321,
            verbose=0,
            n_jobs=n_jobs,
        )
        add_on = df_forecast.forecast.iloc[::-1]
        add_on.index = regressor_train.head(forecast_length).index
        regressor_train = pd.concat(
            [add_on, regressor_train.tail(df_inner.shape[0] - forecast_length)]
        )
    return regressor_train, regressor_forecast
