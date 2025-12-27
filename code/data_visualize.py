import time
from config import START
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from data_loader import data_loader
from data_clean import data_clean
from data_explore import data_explore,split_columns_by_type

plt.rcParams['font.family'] = 'Heiti TC'


def plot_numeric_distribution(df: pd.DataFrame, numeric_cols: list):
    """
    绘制所有数值型特征的分布图
    """
    for col in numeric_cols:
        plt.figure(figsize=(6, 4), dpi=300)
        sns.histplot(df[col], bins=50, kde=True)
        plt.title(f"{col} 分布")
        plt.tight_layout()
        plt.show()


def plot_categorical_distribution(df: pd.DataFrame, categorical_cols: list):
    """
    绘制所有类别型特征的频数分布
    """
    for col in categorical_cols:
        plt.figure(figsize=(6, 4), dpi=300)
        df[col].value_counts().plot(kind="bar")
        plt.title(f"{col} 分布")
        plt.tight_layout()
        plt.show()


def plot_category_vs_numeric(
        df: pd.DataFrame,
        categorical_cols: list,
        numeric_cols: list
):
    """
    类别特征 vs 数值特征 的箱线图
    """
    for cat_col in categorical_cols:
        for num_col in numeric_cols:
            plt.figure(figsize=(6, 4), dpi=300)
            sns.boxplot(x=cat_col, y=num_col, data=df)
            plt.title(f"{num_col} by {cat_col}")
            plt.tight_layout()
            plt.show()


def plot_correlation_heatmap(df: pd.DataFrame):
    """
    绘制数值型特征相关性热力图
    """
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    corr = numeric_df.corr()

    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def data_visualize()-> None:
    """
    数据可视化总入口
    :return:
    """
    numeric_cols, categorical_cols = split_columns_by_type(data_loader())
    data_clean(data_loader(),numeric_cols, categorical_cols)
    print(f'数据可视化开始')
    plot_numeric_distribution(data_loader(), numeric_cols)
    print()

if __name__ == '__main__':
    data_visualize()
    print(f'{time.time() - START:.2f}s')
