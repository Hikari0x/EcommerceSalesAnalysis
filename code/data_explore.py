import time
import pandas as pd

from config import START
from data_loader import load_raw_data


def explore_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    统计各字段缺失率
    :param df:
    :return:
    """
    # 计算每列的缺失值数量
    missing_count = df.isnull().sum()
    # 计算每列的缺失值比率
    missing_rate = missing_count / len(df)
    # print(missing_rate)
    # print(missing_count)
    # 创建包含缺失值数量和缺失率的DataFrame，并按缺失率降序排列
    result = pd.DataFrame({
        'missing_count': missing_count,
        'missing_rate': missing_rate
    }).sort_values(by='missing_rate', ascending=False)
    return result


def explore_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    数值型特征描述性统计
    :param df:原始的数据
    :return:返回数值型特征的描述性统计信息,并转置结果（T）使特征作为行显示
    """
    # 选择数据框中的数值型列（整数和浮点数类型）
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    return numeric_df.describe().T


def explore_categorical_features(df: pd.DataFrame) -> dict:
    """
    类别型特征分布
    :return:返回用户各类别数量
    """
    # 选择数据中的object类型的列
    cat_df = df.select_dtypes(include=['object'])
    result = {}
    for col in cat_df.columns:
        # 统计每列的值分布（包含缺失值）
        result[col] = df[col].value_counts(dropna=False)
    return result


def categorical_by_lifecycle(df, col):
    """
    分析不同生命周期用户中，某类别特征的分布情况
    :param df:
    :param col:
    :return:
    """
    return pd.crosstab(
        df["lifecycle"],
        df[col],
        normalize="index"
    )


if __name__ == '__main__':
    df = load_raw_data()
    result = explore_missing_values(df)
    print(result)
    print(result.shape)
    nd = explore_numeric_features(df)
    print(nd)
    re = explore_categorical_features(df)
    print(re)
    print(categorical_by_lifecycle(df, 'gender'))
    print(f'{time.time() - START:.2f}s')
