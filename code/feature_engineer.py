import time
from config import START
import pandas as pd
from data_loader import load_raw_data
from data_clean import clean_data

def one_hot_encode(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    """
    对类别型特征进行 One-Hot 编码
    只需要对类别列进行翻译
    """
    df_new = df.copy()
    df_new = pd.get_dummies(
        df_new,
        columns=categorical_cols,
        drop_first=False
    )
    return df_new


def build_features(
        df: pd.DataFrame,
        categorical_cols: list
) -> pd.DataFrame:
    """
    特征工程主入口
    """
    df_new = df.copy()
    df_new = one_hot_encode(df_new, categorical_cols)
    return df_new


if __name__ == '__main__':
    df_raw = load_raw_data()
    numeric_cols = ["age", "revenue", "days_since_last_order"]
    categorical_cols = ["gender", "lifecycle"]
    df_clean = clean_data(
        df_raw,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols
    )
    # 3. 特征工程
    df_features = build_features(
        df_clean,
        categorical_cols=categorical_cols
    )
    # 4. 验证结果
    print("原始列数:", df_clean.shape[1])
    print("特征列数:", df_features.shape[1])
    print(df_features.head())
    print(f'{time.time() - START:.2f}s')

