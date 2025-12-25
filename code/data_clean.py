import time

import pandas as pd

from config import START
from data_loader import load_raw_data


def handle_missing_values(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """
    数值型特征缺失值处理：
    - 使用 -1 进行占位填充
    - 同时构造缺失指示变量
    :param df:原始数据
    :param numeric_cols:数值列
    :return:
    """
    df_new = df.copy()
    # 遍历所有数值型列
    for col in numeric_cols:
        # 创建缺失值标志列名
        null_flag_col = f"{col}_is_null"
        # 创建一个新列，标记原始列中哪些值是缺失的（1表示缺失，0表示非缺失）
        df_new[null_flag_col] = df_new[col].isnull().astype(int)
        # 用-1填充原始列中的缺失值
        df_new[col] = df_new[col].fillna(-1)
    return df_new


def handle_categorical_missing(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    """
    类别型特征缺失值处理：填充为 'Unknown'
    """
    df_new = df.copy()
    for col in categorical_cols:
        df_new[col] = df_new[col].fillna("Unknown")
    return df_new


def mark_abnormal_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    异常值标记，不修改原始取值
    """
    df_new = df.copy()
    if "age" in df_new.columns:
        df_new["age_abnormal"] = (
                (df_new["age"] <= 0) | (df_new["age"] > 100)
        ).astype(int)
    return df_new


def clean_data(
        df: pd.DataFrame,
        numeric_cols: list,
        categorical_cols: list
)-> pd.DataFrame:
    """
    统一保存
    :return:
    """
    # print(f'清洗前：{df.shape}\n{df.info()}')
    df_new = df.copy()
    df_new = handle_missing_values(df_new, numeric_cols)
    df_new = handle_categorical_missing(df_new, categorical_cols)
    df_new = mark_abnormal_values(df_new)
    # print(f'清洗后：{df_new.shape}\n{df_new.info()}')

    return df_new


if __name__ == '__main__':
    df = load_raw_data()
    numeric_cols=['age','revenue','days_since_last_order']
    categorical_cols=['gender','lifecycle']
    df_new = clean_data(df, numeric_cols, categorical_cols)
    # print(df_new)
    print(f'{time.time() - START:.2f}s')
