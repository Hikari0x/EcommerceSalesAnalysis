import time

import pandas as pd
from config import RAW_DATA_PATH, START
from pprint import pprint

def load_raw_data():
    """
    加载原始用户数据
    - 读取CSV / 数据库
    - 不做任何清洗与修改
    - 保证“原始性”
    :return:
    """
    df = pd.read_csv(RAW_DATA_PATH)
    # 去除列名中前后的空格,规范化列名
    df.columns=df.columns.str.strip()
    return df


def get_basic_info(df: pd.DataFrame):
    """
     获取数据基本信息
    :param df:传入原始数据
    :return:返回部分数据
    """
    # 获取数据集基本信息
    df.info()
    # 选择部分信息存入字典
    info = {
        "num_samples": df.shape[0],
        "num_features": df.shape[1],
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict()
    }
    return info


if __name__ == '__main__':
    df = load_raw_data()
    info = get_basic_info(df)
    # 打印复杂的结构信息
    pprint(info)
    print(f'{time.time() - START:.2f}s')
