import time
from config import START
import pandas as pd
from data_loader import data_loader
from data_clean import data_clean
from sklearn.preprocessing import StandardScaler, LabelEncoder


def scale_numeric_features(
        df: pd.DataFrame,
        numeric_cols: list,
        scaler: StandardScaler | None = None
):
    """
    数值特征标准化
    :param df:DataFrame清洗好的数据集
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
        numeric_cols: list,
        categorical_cols: list,
        scaler: StandardScaler | None = None
):
    """
    深度学习模型特征工程主入口
    特点：
    - 数值特征：标准化
    - 类别特征：整数编码（Embedding 用）
    - 不做 One-Hot
    :param df:DataFrame清洗好的数据集
    - numeric_cols:数值型列名
    - categorical_cols:类别型列名
    :return:
        df_new:DataFrame编码后的数据
        scaler:数值特征标准化器
    """
    df_new = df.copy()
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
