import time

import pandas as pd

import config
import data_loader
import data_explore


def handle_missing_values(
        df: pd.DataFrame,
        numeric_cols: list,
        fill_value: int = -1
) -> pd.DataFrame:
    """
    数值型特征缺失值处理：
    1. 使用指定值（默认 -1）填充缺失
    2. 为每个数值特征创建缺失指示变量
    :param df:原始 DataFrame数据集
    :param numeric_cols:数值列
    :return: 填充后的数据
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


def handle_categorical_missing(df: pd.DataFrame, categorical_cols: list, fill_value: str = 'Unknown') -> pd.DataFrame:
    """
    类别型特征缺失值处理：填充为 'Unknown'
    :param df:原始数据集
    :param categorical_cols:类别型列
    :param fill_value:填充值，默认为 'Unknown'
    :return:填充后的数据
    """
    df_new = df.copy()
    for col in categorical_cols:
        df_new[col] = df_new[col].fillna(fill_value)
    return df_new


def mark_abnormal_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    异常值标记（不修改原始数值）
    当前仅对 age 进行简单规则标记
    - age <= 0 或 age > 100 视为异常
    :param df: DataFrame
    :return: 添加异常标记列后的 DataFrame数据集
    """
    df_new = df.copy()
    if "age" in df_new.columns:
        df_new["age_abnormal"] = (
                (df_new["age"] <= 0) | (df_new["age"] > 100)
        ).astype(int)
    return df_new


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    重复值处理
    - 统计重复行数量
    - 删除完全重复的行（保留第一条）
    :param df: DataFrame
    :return: 去重后的 DataFrame数据集
    """
    # 统计重复行数量
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        print(f"发现重复行数量：{dup_count}，已执行删除")
        # 删除重复行，保留第一次出现的
        df_new = df.drop_duplicates(keep="first")
    else:
        print("未发现重复行")
    return df_new


def save_clean_data(df: pd.DataFrame, path: str) -> None:
    """
    保存清洗后的数据
    :param df: DataFrame
    :param path: 保存路径
    """
    df.to_csv(path, index=False)
    print(f"清洗后数据已保存至：{path}")


def data_clean(
        df: pd.DataFrame,
        numeric_cols: list,
        categorical_cols: list
) -> pd.DataFrame:
    """
    数据清洗主函数
    :return:清洗好的新数据集df_new
    """
    # print(f'清洗前：{df.shape}\n{df.info()}')
    df_new = df.copy()
    df_new = handle_missing_values(df_new, numeric_cols)
    df_new = handle_categorical_missing(df_new, categorical_cols)
    df_new = mark_abnormal_values(df_new)
    df_new = remove_duplicates(df_new)
    # print(f'清洗后：{df_new.shape}\n{df_new.info()}')
    save_clean_data(df_new, config.DF_NEW_PATH)
    return df_new

if __name__ == '__main__':
    numeric_cols, categorical_cols = data_explore.split_columns_by_type(data_loader.load_raw_data())
    data_clean(data_loader.load_raw_data(), numeric_cols, categorical_cols)
    print(f'{time.time() - config.START:.2f}s')
