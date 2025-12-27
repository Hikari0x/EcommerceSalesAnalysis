import time
from config import START, TARGET_COL
import pandas as pd
from data_loader import data_loader
from data_clean import data_clean
from sklearn.preprocessing import StandardScaler, LabelEncoder


def add_missing_indicators(
        df: pd.DataFrame,
        numeric_cols: list,
        missing_value: int | float = -1
) -> pd.DataFrame:
    """
    为数值型特征添加缺失值指示变量
    :param df:DataFrame清洗好的数据集
    :param numeric_cols:需要处理缺失值的数值型列名列表
    :param fill_value:表示缺失值的占位值，默认是 -1
    :return: 处理后的
    """
    df_new = df.copy()
    for col in numeric_cols:
        df_new[f"{col}_is_missing"] = (df_new[col] == missing_value).astype(int)

    return df_new


def split_columns_by_type(df: pd.DataFrame, target_col: str):
    """
    自动划分需要标准化的列,数值列和缺失值指示异常列
    :param df: DataFrame添加指示变量的数据
    :return:
        numeric_cols: 标准化列
        categorical_cols: 类别列
        indicator_cols: 缺失值指示列和异常列
    """
    # 1. 初步按 dtype 划分
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # 2. 从类别特征中剔除标签列
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    # 3. 识别缺失值指示 & 异常标记列
    indicator_cols = [
        col for col in numeric_cols
        if col.endswith("_is_null") or col.endswith("_abnormal")
    ]
    # 4. 真正用于数值建模的列（需要标准化）
    numeric_cols = [
        col for col in numeric_cols
        if col not in indicator_cols
    ]

    return numeric_cols, categorical_cols, indicator_cols


def scale_numeric_features(
        df: pd.DataFrame,
        numeric_cols: list,
        scaler: StandardScaler | None = None
):
    """
    数值特征标准化
    :param df:DataFrame
    :param numeric_cols:数值型列名
    :param scaler:训练阶段传 None，预测阶段传已有 scaler
    :return:
        df_new:DataFrame标准化后的数据
        scaler:
    """
    df_new = df.copy()
    if scaler is None:
        scaler = StandardScaler()
        df_new[numeric_cols] = scaler.fit_transform(df_new[numeric_cols])
    else:
        df_new[numeric_cols] = scaler.transform(df_new[numeric_cols])

    return df_new, scaler


def one_hot_encode(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    """

    :param df:DataFrame清洗好的数据集
    :param categorical_cols:类别型列名
    :return:One-Hot 编码后的数据
    """

    return pd.get_dummies(df, columns=categorical_cols, drop_first=False)


def build_features_for_ml(
        df: pd.DataFrame,
        numeric_cols: list,
        categorical_cols: list,
        scaler: StandardScaler | None = None
):
    """
    传统机器学习模型（LR / RF）特征工程主入口
    步骤：
    1. 数值特征标准化
    2. 类别特征 One-Hot 编码
    :param df:
    :param numeric_cols:
    :param categorical_cols:
    :param scaler:
    :return:
        df_new:DataFrame最终可用于模型训练的数据
        scaler:数值特征标准化器
    """
    df_new = df.copy()
    # 1. 数值特征标准化
    df_new, scaler = scale_numeric_features(
        df_new,
        numeric_cols=numeric_cols,
        scaler=scaler
    )
    # 2. 类别特征 One-Hot
    df_new = one_hot_encode(
        df_new,
        categorical_cols=categorical_cols
    )

    return df_new, scaler


def encode_categorical_for_dl(df: pd.DataFrame, categorical_cols: list):
    """
    深度学习用的类别特征编码
    - 每个类别列编码成整数
    - 后续交给 Embedding 层
    :param df:DataFrame清洗好的数据集
    :param categorical_cols:类别型列名
    :return:
        df_new:DataFrame编码后的数据
        encoders:编码器
    """
    df_new = df.copy()
    encoders = {}
    for col in categorical_cols:
        encoder = LabelEncoder()
        df_new[col] = encoder.fit_transform(df_new[col].astype(str))
        encoders[col] = encoder

    return df_new, encoders


def build_features_for_dl(
        df: pd.DataFrame,
        scaler: StandardScaler | None = None
):
    """
    深度学习模型特征工程主入口
    特点：
    - 数值特征：标准化
    - 类别特征：整数编码（Embedding 用）
    - 不做 One-Hot
    :param df:DataFrame清洗好的数据集
    :return:
        df_new:DataFrame编码后的数据
        scaler:数值特征标准化器
        encoders:编码器
    """
    df_new = df.copy()
    numeric_cols, categorical_cols, indicator_cols = split_columns_by_type(df_new, TARGET_COL)
    df_new = add_missing_indicators(df_new, numeric_cols, missing_value=-1)
    # 1. 数值特征标准化
    df_new, scaler = scale_numeric_features(
        df_new,
        numeric_cols=numeric_cols,
        scaler=scaler
    )
    # 2. 类别特征整数编码
    df_new, encoders = encode_categorical_for_dl(
        df_new,
        categorical_cols=categorical_cols
    )

    return df_new, scaler, encoders


if __name__ == '__main__':
    print(f'{time.time() - START:.2f}s')
