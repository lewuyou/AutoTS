'''股票相关数据处理函数'''
import pandas as pd

def data_to_df(data,w,n):
    '''拼接final_data的数据*n'''
    data = pd.DataFrame(data[i*w:(i+n)*w] for i in range(int(len(data)/w)-n+1)) #将一维数组转换成dataframe,每行的数据是data的第i*w个数据到第(i+n)*w个数据
    return data

def del_col(data,w,n):
    '''清理不需要的feat标签列'''
    for i in range(1,n):
        data = data.drop([(i)*w-1],axis=1) # 删除第(i)*w-1列，直到删除完第n列
    return data

def split_xy(data):
    '''将数据拆分出x和y'''
    y =  data[:,-1] # 最后一列是y
    x = data[:,:-1] # 剩下的是x
    return x, y

def twf_feat(close, high, low, valume,period=21):
    ''' 计算twf'''
    data = pd.DataFrame()
    data['Close'] = close
    data['High'] = high
    data['Low'] = low
    data['Volume'] = valume
    data['Close_yday'] = data['Close'].shift(1) # 昨天收盘价
    data['tr_h_max'] = data[['High', 'Close_yday']].max(axis=1) # 计算今天最高价和昨天收盘价的最大值
    data['tr_l_min'] = data[['Low', 'Close_yday']].min(axis=1) # 计算今天最低价和昨天收盘价的最小值
    data['tr_c'] = data['tr_h_max'] - data['tr_l_min'] # 计算真实波动幅度
    data['tr_tmp1'] = data['Close'] - data['tr_l_min'] # 计算Close减去tr_l_min的值
    data['tr_tmp2'] = data['tr_h_max'] - data['Close'] # 计算tr_h_max减去Close的值
    data['adv'] = data['Volume'] * (data['tr_tmp1'] - data['tr_tmp2']) / data['tr_c'].replace(0, 99999999) # 计算Volume*(tr_tmp1-tr_tmp2)/tr_c的值,如果tr_c为0，那就用99999999代替
    data['vol_ma'] = 0.0
    data['adv_ma'] = 0.0
    for i in range(1, len(data)):
        data.loc[[i],['vol_ma']] = data['Volume'] + data['vol_ma'].shift(1)*(period-1)/period
    for i in range(1, len(data)):
        data.loc[[i],['adv_ma']] = data['adv'] + data['adv_ma'].shift(1)*(period-1)/period
    data['twf'] = data['adv_ma'] / data['vol_ma'] # 计算twf
    return data['twf'].values