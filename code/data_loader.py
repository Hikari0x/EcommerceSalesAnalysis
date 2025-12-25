import time

import pandas as pd
from config import RAW_DATA_PATH,START


def load_raw_data():
    """
    加载原始用户数据
    - 读取CSV / 数据库
    - 不做任何清洗与修改
    - 保证“原始性”
    :return:
    """
    df = pd.read_csv(RAW_DATA_PATH)
    return df


def get_basic_info(df: pd.DataFrame) -> dict:
    """
     获取数据基本信息
    :param df:传入原始数据
    :return:基本数据封装到字典并返回
    """
    info = {
        'num_samples': df.shape[0],
        'num_features': df.shape[1],
        'columns': list(df.columns)
    }
    return info


if __name__ == '__main__':
    df = load_raw_data()
    info = get_basic_info(df)
    print(info)
    print(f'{time.time()-START:.2f}s')
