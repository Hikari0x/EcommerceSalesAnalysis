import time
import pandas as pd

from config import RAW_DATA_PATH, START


def explore_missing_values(df: pd.DataFrame)->pd.DataFrame:
    """
    统计各字段缺失率
    :param df:
    :return:
    """


def explore_basic_stats(df):
    pass


if __name__ == '__main__':
    print(f'{time.time() - START:.2f}s')
