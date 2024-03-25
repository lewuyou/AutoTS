# create_data.py 创建数据集
import akshare as ak
import datetime
from data_provider.func_util import *
from data_provider.func_stock import *
from data_provider.func_RankGauss import *
from data_provider.ichimoku_cloud import *

from pyti.chaikin_money_flow import chaikin_money_flow as cmf
from pyti.moving_average_convergence_divergence import moving_average_convergence_divergence as macd
from pyti.relative_strength_index import  relative_strength_index as rsi
from pyti.stochastic import percent_k as kdj_k
from pyti.stochastic import percent_d as kdj_d
from pyti.hull_moving_average import hull_moving_average as hma
from pyti.exponential_moving_average import exponential_moving_average as ema
from pyti.bollinger_bands import *
from pyti.commodity_channel_index import commodity_channel_index as cci
from pyti.williams_percent_r import williams_percent_r as wr
from pyti.on_balance_volume import on_balance_volume as obv
from pyti.accumulation_distribution import accumulation_distribution as acc_dist
from pyti.chaikin_money_flow import chaikin_money_flow as cmf2

def download_data(num,args):
    '''下载指定时间段数据'''
    if args.start_date == 1:
        args.num_start_data = ak.stock_individual_info_em(symbol=num).iat[3,1]    # 开始时间选取公司信息里面的上市时间
    else:
        args.num_start_data = args.start_date
    if args.end_date == 1:
        args.num_end_data = (datetime.datetime.now()).strftime("%Y%m%d")   # 结束选择今天
    elif args.end_date == -1:
        args.num_end_data = (datetime.datetime.now()-datetime.timedelta(days=1)).strftime("%Y%m%d")    # 结束选择昨天
    else:
        args.num_end_data = args.end_date
    df_colum = ak.stock_zh_a_hist(symbol=num, period=args.period, start_date= args.num_start_data, end_date= args.num_end_data, adjust=args.fuquan)
    df_resault = df_colum[['日期', '开盘', '收盘', '最高', '最低', '成交量', '振幅', '涨跌幅', '换手率']].copy()    # 只获取其中需要的数据
    df_resault.columns = ['date', 'Open', 'Close', 'High', 'Low', 'Volume', 'Amplitude', 'Change', 'Turnover']    # 重命名表头
    print(f'获取数据时间为： {args.num_start_data} - {args.num_end_data}')
    print(f'原始数据形状： {df_resault.shape}')
    return df_resault


def add_data(raw_data,args):
    '''创建模型合适的数据集train_loader, valid_loader, test_loader'''

    # 提取列数据为NumPy数组
    open_data = raw_data['Open'].values
    close_data = raw_data['Close'].values
    high_data = raw_data['High'].values
    low_data = raw_data['Low'].values
    volume_data = raw_data['Volume'].values
    
    # 归一化 成交量 到 0-10 范围
    min_volume = volume_data.min()
    max_volume = volume_data.max()
    raw_data['K_volume'] = 10 * (volume_data - min_volume) / (max_volume - min_volume)

    '''趋势指标'''
    # MACD
    raw_data['macd'] = macd(close_data, 12, 26) # default: fast_period=12, slow_period=26, signal_period=9
    raw_data['macd2'] = macd(close_data, 48, 104)
    # 移动平均线
    raw_data['hma10'] = hma(close_data, 10)
    raw_data['hma20'] = hma(close_data, 20)
    raw_data['hma60'] = hma(close_data, 60)
    # raw_data['ema10'] = ema(close_data, 10)
    # raw_data['ema20'] = ema(close_data, 20)
    # raw_data['ema60'] = ema(close_data, 60)
    # 布林带指标
    raw_data['upboll'] = upper_bollinger_band(close_data, 20)
    raw_data['miboll'] = middle_bollinger_band(close_data, 20)
    raw_data['loboll'] = lower_bollinger_band(close_data, 20)
    raw_data['baboll'] = bandwidth(close_data, 20)
    raw_data['bbboll'] = bb_range(close_data, 20)
    raw_data['peboll'] = percent_bandwidth(close_data, 20)
    raw_data['pbboll'] = percent_b(close_data, 20)
    # 一目均衡表
    raw_data['ichimoku1'] = tenkansen(close_data)
    raw_data['ichimoku2'] = kijunsen(close_data)
    raw_data['ichimoku3'] = chiku_span(close_data)
    raw_data['ichimoku4'] = senkou_a(close_data)
    raw_data['ichimoku5'] = senkou_b(close_data)
    
    '''震荡指标'''
    # Rsi  相对强弱指标
    raw_data['rsi'] = rsi(close_data, 14)
    # KDJ 随机指标
    raw_data['kdj_k'] = kdj_k(close_data, 9)
    raw_data['kdj_d'] = kdj_d(close_data, 9)
    # cci 用来衡量股价是否已经偏离其平均价格
    raw_data['cci'] = cci(close_data, high_data, low_data, 20)
    # Williams %R 威廉指标
    raw_data['wr'] = wr(close_data)
    
    '''成交量指标'''
    # Chaikin Money Flow 蔡金货币流量指标
    raw_data['twf_feat'] = twf_feat(close_data, high_data, low_data, volume_data, 21)
    # obv
    # raw_data['obv'] = obv(close_data, volume_data)
    # acc_dist 累积/派发指标
    # raw_data['acc_dist'] = acc_dist(close_data, high_data, low_data, volume_data)
    # cmf2
    # raw_data['cmf2'] = cmf2(close_data, high_data, low_data, volume_data, 20)
    
    
    print(f'添加数据以后形状： {raw_data.shape}')
    return raw_data

def add_label(data, args):
    '''原始数据计算并添加预测标签'''
    data['Tom_Chg'] = (data['Close'].shift(-args.label_n) - data['Close'])/ data['Close']# 计算第n天到今天收益率
    if args.label_ch:
        # 初始化OT列为0
        data['OT'] = 0
        # 如果第n天到今天收益率data['Tom_Chg']大于0.5，那么label就等于1
        data.loc[data['Tom_Chg'] >= args.zhangfu, 'OT'] = 1
    else:
        data['OT'] = data['Tom_Chg']
    data = data.fillna(0)#将数据中的nan替换为0
    print(f'添加label以后数据形状: {data.shape}')
    return data

def sub_data(data,args):
    # 删除前24行（macd算不出来）和最后的n行（最后n行未来涨跌幅是0）
    # 根据 args.end 的值来选择正确的切片方法
    if args.end == 0:
        label_data = data.iloc[args.start:, :]
    else:
        label_data = data.iloc[args.start:-args.end, :]

    # 指定删除不需要列
    final_data = label_data.drop(columns=args.final_data_feat, errors='ignore')
    print('删除指定行、列后数据形状: ',final_data.shape)
    return final_data

def add_zeros_to_data(data, num_rows=20):
    """
    在数据的前num_rows行添加0,但保留第一列不变。

    :param data: DataFrame,需要添加0的数据。
    :param num_rows: int,需要添加0的行数。
    :return: DataFrame,已添加0的数据。
    """
    data_copy = data.copy()
    data_copy.iloc[:num_rows, 1:] = 0  # 除第一列外，前num_rows行替换为0
    return data_copy