import time

import numpy as np
import pandas as pd

import config
import data_loader


def explore_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    统计各字段缺失率
    :return:DataFrame，包含每个字段的缺失数量和缺失率，按缺失率降序排列
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
    # 添加缺失值模式分析
    missing_pattern = []
    for col in df.columns:
        if missing_rate[col] > 0:
            if missing_rate[col] > 0.5:
                pattern = "大量缺失"
            elif missing_rate[col] > 0.2:
                pattern = "中等缺失"
            else:
                pattern = "少量缺失"
        else:
            pattern = "无缺失"
        missing_pattern.append(pattern)
    result['missing_pattern'] = missing_pattern
    return result


def explore_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    分析数值型特征的描述性统计
    :param df:输入原始的数据集
    :return:DataFrame，包含各数值型特征的统计量（计数、均值、标准差、最小值、25%、50%、75%、最大值）
    """
    # 选择数据框中的数值型列（整数和浮点数类型）
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    return numeric_df.describe().T


def explore_categorical_features(df: pd.DataFrame, top_n: int = 10) -> dict:
    """
    分析类别型特征的分布情况
    :param df:输入原始的数据集
    :param top_n:int，显示每个类别的前N个值，默认显示前10个
    :return:字典，键为类别型列名，值为该列的值分布Series
    """
    result = {}
    # 识别类别型列（字符串类型或分类类型）
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    for col in cat_cols:
        # 统计每个值的出现次数（包括NaN）
        value_counts = df[col].value_counts(dropna=False)
        result[col] = value_counts.head(top_n)
    return result


def analyze_feature_by_group(
        df: pd.DataFrame,
        group_col: str,
        feature_col: str,
        normalize: bool = True
) -> pd.DataFrame:
    """
    分析某个特征在不同分组中的分布
    :param df: pandas DataFrame，输入的数据集
    :param group_col:str，分组列名（如'lifecycle'）
    :param feature_col:str，要分析的特征列名
    :param normalize:bool，是否计算比例而不是计数，默认为True（计算比例）
    :return:DataFrame，交叉表显示特征在不同分组中的分布
    """
    if normalize:
        return pd.crosstab(df[group_col], df[feature_col], normalize='index')
    else:
        return pd.crosstab(df[group_col], df[feature_col])


def explore_correlation(df: pd.DataFrame, method: str = 'pearson', threshold: float = 0.7) -> pd.DataFrame:
    """
    分析数值特征之间的相关性
    :param df:DataFrame，原始数据
    :param method: 相关系数计算方法 ('pearson', 'spearman', 'kendall'),默认 pearson
    :param threshold: 相关性阈值，用于筛选强相关特征对
    :return:
        corr_matrix : 相关性矩阵
        strong_corr : 强相关特征对（DataFrame）
    """
    # 1. 只保留数值型列（int、float）
    numeric_df = df.select_dtypes(include=[np.number])
    # 2. 计算相关性矩阵
    corr_matrix = numeric_df.corr(method=method)
    # 3. 用列表保存强相关结果
    strong_corr_list = []
    # 4. 两层 for 循环，逐个比较特征之间的相关性
    columns = corr_matrix.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            feature_1 = columns[i]
            feature_2 = columns[j]
            corr_value = corr_matrix.loc[feature_1, feature_2]
            # 5. 如果相关性绝对值大于阈值，就保存
            if abs(corr_value) >= threshold:
                strong_corr_list.append([
                    feature_1,
                    feature_2,
                    corr_value,
                    abs(corr_value)
                ])
    # 6. 转成 DataFrame，方便查看
    strong_corr = pd.DataFrame(
        strong_corr_list,
        columns=['feature1', 'feature2', 'correlation', 'abs_correlation']
    )
    # 7. 按绝对相关性从大到小排序
    strong_corr = strong_corr.sort_values(
        by='abs_correlation',
        ascending=False
    )
    return corr_matrix, strong_corr


def split_columns_clean(df: pd.DataFrame):
    """
    用于清洗数据划分列
    :param df: DataFrame原始数据
    :return: 数值列和类别列
    """
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    return numeric_cols, categorical_cols


def data_explore():
    """
    展示所有探索的数据
    :return:
    """
    df = data_loader.load_raw_data()
    print(f'统计各字段缺失率')
    re = explore_missing_values(df)
    print(re)
    print(f'分析数值型特征的描述性统计')
    nd = explore_numeric_features(df)
    print(nd)
    print(f'分析类别型特征的分布情况')
    result = explore_categorical_features(df)
    print(result)
    print(f'分析某个特征在不同分组中的分布')
    group = analyze_feature_by_group(df, 'lifecycle', 'age')
    print(group)
    print(f'分析数值特征之间的相关性')
    correlation = explore_correlation(df, 'pearson', 0.7)
    print(correlation)
    print(f'自动划分数值列和类别列')
    numeric_cols, categorical_cols = split_columns_clean(df)
    print(f'建模列:{numeric_cols}'
          f'类别列:{categorical_cols}')


if __name__ == '__main__':
    data_explore()
    print(f'{time.time() - config.START:.2f}s')
