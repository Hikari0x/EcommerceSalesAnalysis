import time
from config import START
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from data_loader import load_raw_data
from data_clean import clean_data
from data_explore import categorical_by_lifecycle

plt.rcParams['font.family'] = 'Heiti TC'


def plot_numeric_distribution(df: pd.DataFrame, col: str):
    """
    数值型特征分布
    """
    plt.figure(figsize=(6, 4), dpi=300)
    sns.histplot(df[col], bins=50, kde=True)
    plt.title(f'{col}分布')
    plt.show()


def plot_categorical_distribution(df: pd.DataFrame, col: str):
    """
    类别型特征分布
    """
    plt.figure(figsize=(6, 4), dpi=300)
    df[col].value_counts().plot(kind='bar')
    plt.title(f"Distribution of {col}")
    plt.show()


def plot_box_by_category(df: pd.DataFrame, cat_col: str, num_col: str):
    """
    类别-数值关系箱线图
    """
    plt.figure(figsize=(6, 4), dpi=300)
    sns.boxplot(x=cat_col, y=num_col, data=df)
    plt.title(f"{num_col} by {cat_col}")
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame):
    """
    数值特征相关性热力图
    """
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    corr = numeric_df.corr()

    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()


def plot_categorical_by_lifecycle(df, col):
    table = categorical_by_lifecycle(df, col)
    table.plot(kind="bar", stacked=True)
    plt.show()


if __name__ == '__main__':
    df = load_raw_data()
    numeric_cols = ['age', 'revenue', 'days_since_last_order']
    categorical_cols = ['gender', 'lifecycle']
    df_new = clean_data(df, numeric_cols, categorical_cols)
    # 数值分布
    plot_numeric_distribution(df_new, 'age')
    # 类别分布
    plot_categorical_distribution(df_new, 'gender')
    # 类别-数值关系
    plot_box_by_category(df_new, 'gender', 'age')
    # 相关性
    plot_correlation_heatmap(df_new)
    plot_categorical_by_lifecycle(df_new, 'gender')
    print(f'{time.time() - START:.2f}s')
