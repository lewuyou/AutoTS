import torch 
import numpy as np

def same_seeds(seed):
    torch.manual_seed(seed) # 固定cpu的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # 固定GPU的随机种子
        torch.cuda.manual_seed_all(seed)  # 固定多个GPU的随机种子
    np.random.seed(seed)  # 固定numpy的随机种子
    torch.backends.cudnn.benchmark = False # 卷积层进行预先的优化 关闭
    torch.backends.cudnn.deterministic = True # 相同的卷积算法

def normalize(data):
    '''归一化数据'''
    data = data.astype(np.float32) # 转换数据类型
    _range = np.max(data) - np.min(data) # 计算极差
    return (data - np.min(data)) / _range # 归一化

def Standardization(data):
    '''标准化数据'''
    df_mean = np.mean(data)
    df_std = np.std(data)
    alias_train = (data - df_mean) / df_std
    return alias_train

def QTF(data):
    '''使用sklearn的QuantileTransformer进行转换'''
    from sklearn.preprocessing import QuantileTransformer
    qt = QuantileTransformer(n_quantiles=100, random_state=0,output_distribution='normal')
    data = np.array(data)
    data = data.reshape(-1, 1)
    data = qt.fit_transform(data)
    # 将数据转换成一维数据
    data = data.reshape(1, -1)[0]
    return data

def RobustScaler(data):
    '''使用sklearn的RobustScaler进行转换'''
    from sklearn.preprocessing import RobustScaler
    data = np.array(data)
    data = data.reshape(-1, 1)
    data = RobustScaler().fit_transform(data)
    # 将数据转换成一维数据
    data = data.reshape(1, -1)[0]
    return data

def PowerTransformer(data):
    '''使用sklearn的PowerTransformer进行转换'''
    from sklearn.preprocessing import PowerTransformer
    data = np.array(data)
    data = data.reshape(-1, 1)
    data = PowerTransformer().fit_transform(data)
    # 将数据转换成一维数据
    data = data.reshape(1, -1)[0]
    return data

def StandardScaler(data):
    '''使用sklearn的StandardScaler进行转换'''
    from sklearn.preprocessing import StandardScaler
    data = np.array(data)
    data = data.reshape(-1, 1)
    data = StandardScaler().fit_transform(data)
    # 将数据转换成一维数据
    data = data.reshape(1, -1)[0]
    return data

def valid_split(data, test_size, random_state):
    from sklearn.model_selection import StratifiedShuffleSplit
    '''使用StratifiedShuffleSplit拆分数据集'''
    data = np.array(data) # 将数据转换为numpy数组,否则会报错
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_index, test_index in sss.split(data, data[:,-1]):
        train_data, test_data = data[train_index], data[test_index]
    print(f'valid_split:\n分层随机拆分 train_data_all: {len(data)},\ntrain_data: {len(train_data)},\nvalid_data: {len(test_data)}')
    return train_data, test_data


def train_split(data, test_size, random_state):
    from sklearn.model_selection import ShuffleSplit
    '''使用ShuffleSplit拆分数据集'''
    data = np.array(data) # 将数据转换为numpy数组,否则会报错
    ss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_index, test_index in ss.split(data):
        train_data, test_data = data[train_index], data[test_index]
    print(f'“ShuffleSplit随机”拆分data_all: {data.shape}: \n拆分后Train训练集 {train_data.shape} , Valid验证集 {test_data.shape}')
    return train_data, test_data

def time_split(data, n_splits):
    from sklearn.model_selection import TimeSeriesSplit
    '''使用TimeSeriesSplit拆分数据集'''
    data = np.array(data) # 将数据转换为numpy数组,否则会报错
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_index, test_index in tscv.split(data):
        train_data, test_data = data[train_index], data[test_index]
    print(f'TimeSeriesSplit时间序列拆分data_all数据{n_splits}次： \n拆分后Train训练集 {train_data.shape} , Valid验证集 {test_data.shape}')
    return train_data, test_data

def kfold_split(data, n_splits, random_state):
    from sklearn.model_selection import KFold
    '''使用K折交叉验证拆分数据集'''
    data = np.array(data) # 将数据转换为numpy数组,否则会报错
    kf = KFold(n_splits=n_splits, shuffle=False) # 如果shuffle为True，可以设置种子, random_state=random_state
    for train_index, test_index in kf.split(data):
        # print(f"TRAIN:\n{train_index}, \nTEST:\n{test_index}" )#获得索引值
        train_data, test_data = data[train_index], data[test_index]
    print(f'KFold K折交叉验证拆分data_all数据{n_splits}次： \n拆分后Train训练集 {train_data.shape} , Valid验证集 {test_data.shape}')
    return train_data, test_data

def test_split(data,ratio):
    '''拆分连续的训练集和测试集'''
    train_end = -int(ratio * len(data))
    train_data, test_data = data[:train_end], data[train_end:]
    print(f'“顺序”拆分data: {data.shape}: \n拆分后data1 {train_data.shape} , data2 {test_data.shape}')
    return train_data,test_data

def data_flatten(data):
    '''将数据展开成一维'''
    data = data.values.reshape(-1)
    return data

def multivariate_data(final_data, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    
    if end_index is None:
        end_index = len(final_data) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step) # step表示滑动步长
        data.append(final_data[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)

def update_path(NUM, config):
    '''更新配置文件'''
    config['save_path'] =   f'./models/{NUM}_mode_bat{config["batch_size"]}_hid{config["hidden_dim"]}_sp{config["n_splits"]}_N{config["n"]}_lN{config["lable_n"]}_LR{config["learning_rate"]}_FN{config["feat_normalize"]}.ckpt'
    config['predict_path'] =  f'./data/{NUM}_pred_bat{config["batch_size"]}_hid{config["hidden_dim"]}_sp{config["n_splits"]}_N{config["n"]}_lN{config["lable_n"]}_LR{config["learning_rate"]}_FN{config["feat_normalize"]}.csv'
    config['test_path'] =     f'./data/{NUM}_test_bat{config["batch_size"]}_hid{config["hidden_dim"]}_sp{config["n_splits"]}_N{config["n"]}_lN{config["lable_n"]}_LR{config["learning_rate"]}_FN{config["feat_normalize"]}.csv'
    config['res_acc_path'] =   f'./data/0_all_acc_bat{config["batch_size"]}_hid{config["hidden_dim"]}_sp{config["n_splits"]}_N{config["n"]}_lN{config["lable_n"]}_LR{config["learning_rate"]}_FN{config["feat_normalize"]}.csv'
    return config