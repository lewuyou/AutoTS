"""
FRED（美联储经济数据）数据导入

需要 FRED 的 API 密钥
d84151f6309da8996e4f7627d6efc026 
并 pip install fredapi
"""

import time
import pandas as pd

try:
    from autots.datasets.fred2 import Fred
except Exception:  # except ImportError
    _has_fred = False
else:
    _has_fred = True


def get_fred_data(
    fredkey: str,
    SeriesNameDict: dict = None,
    long=True,
    observation_start=None,
    sleep_seconds: int = 1,
    **kwargs,
):
    """从美联储导入数据。
     为了获得最简单的结果，请确保请求的系列都具有相同的频率。

     参数：
         fredkey (str)：FRED 的 API 密钥
         SeriesNameDict (dict)：成对的 FRED 系列 ID 和系列名称，例如：{'SeriesID': 'SeriesName'} 或 FRED ID 列表。
             系列 ID 必须与 Fred ID 匹配，但名称可以是任何内容
             如果没有，则返回几个默认系列
         long (bool)：如果为 True，则返回长样式数据，否则返回带有 dt 索引的宽样式数据
         Observation_start（日期时间）：传递给 Fred get_series
         sleep_seconds (int): 每个系列调用之间休眠的秒数，通常会减少失败机会
    """
    if not _has_fred:
        raise ImportError("Package fredapi is required")

    fred = Fred(api_key=fredkey)

    if SeriesNameDict is None:
        SeriesNameDict = {
            'T10Y2Y': '这是美国10年期国债与2年期国债收益率之差，通常被用作衡量经济前景的一个指标，特别是在预测经济衰退方面',
            'DGS10': '美国10年期国债的收益率，常用于衡量长期利率水平。',
            'DCOILWTICO': '这个数据代表了德克萨斯州库欣地区西德克萨斯中质原油（WTI）的价格，是油价的一个重要基准',
            'SP500': '标准普尔500指数，是一个美国股市的重要指数，反映了美国大型上市公司的股价表现',
            'DEXUSEU': '美元/欧元汇率，反映了美元对欧元的价值',
            'DEXCHUS': 'China US Foreign Exchange Rate',
            'DEXCAUS': '加拿大元对美元每日汇率',
            'VIXCLS': '芝加哥期权交易所波动率指数：VIX',  # this is a more irregular series
            'T10YIE': '10 年期国债的预期通胀率，通过市场价格推算出来，反映了市场对未来5年平均通胀率的预期',
            'USEPUINDXD': '美国经济政策不确定性指数',  # also very irregular
        }

    if isinstance(SeriesNameDict, dict):
        series_desired = list(SeriesNameDict.keys())
    else:
        series_desired = list(SeriesNameDict)

    if long:
        fred_timeseries = pd.DataFrame(
            columns=['date', 'value', 'series_id', 'series_name']
        )
    else:
        fred_timeseries = pd.DataFrame()

    for series in series_desired:
        print(f"正在下载 {series}...")
        data = fred.get_series(series, observation_start=observation_start)
        try:
            series_name = SeriesNameDict[series]
        except Exception:
            series_name = series

        if long:
            data_df = pd.DataFrame(
                {
                    'date': data.index,
                    'value': data,
                    'series_id': series,
                    'series_name': series_name,
                }
            )
            data_df.reset_index(drop=True, inplace=True)
            fred_timeseries = pd.concat(
                [fred_timeseries, data_df], axis=0, ignore_index=True
            )
        else:
            data.name = series_name
            fred_timeseries = fred_timeseries.merge(
                data, how="outer", left_index=True, right_index=True
            )
        time.sleep(sleep_seconds)

    return fred_timeseries
