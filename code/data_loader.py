import time

import pandas as pd

import config
from pprint import pprint
from sqlalchemy import create_engine


def get_mysql_engine():
    """
    获取数据库连接
    :return:
    """
    engine = create_engine(
        "mysql+pymysql://root:12345678@localhost:3307/ecommerce_sales_analysis?charset=utf8mb4"
    )
    return engine


def table_has_data(engine, table_name: str) -> bool:
    """

    :param engine:
    :param table_name:
    :return:
    """
    sql = f"SELECT COUNT(*) FROM {table_name}"
    result = pd.read_sql(sql, engine)
    return result.iloc[0, 0] > 0


def save_to_mysql(
        df: pd.DataFrame,
        table_name: str,
        engine
):
    """
    load_raw_data已经清洗过列名,去除空格,可以正常存入
    先使用if下面的代码导入数据,然后使用if判断数据没问题
    :param df:原始数据
    :param table_name:提前设定好的表名
    :param engine:数据库
    """
    # 判断表是否为空
    if table_has_data(engine, table_name):
        print(f"{table_name} 已有数据，跳过写入")
        return
    # 写入数据
    df.to_sql(
        name=table_name,
        con=engine,
        if_exists="append",
        index=False
    )


def load_from_mysql(
        engine,
        table_name: str
) -> pd.DataFrame:
    """
    从 MySQL 读取整张表数据
    :param engine: 数据库
    :param table_name: 表名
    :return: DataFrame原始数据集
    """
    sql = f"SELECT * FROM `{table_name}`"
    df = pd.read_sql(sql, engine)
    return df


def load_raw_data() -> pd.DataFrame:
    """
    加载原始用户数据,并规范列名
    - 读取CSV / 数据库
    - 不做任何清洗与修改
    - 保证“原始性”
    :return:返回加载好的原始数据
    """
    df = pd.read_csv(config.RAW_DATA_PATH)
    # 去除列名中前后的空格,规范化列名
    df.columns = df.columns.str.strip()
    return df


def get_basic_info() -> dict:
    """
     获取数据基本信息
    :param df:传入原始数据
    :return:返回部分数据
    """
    # 获取原始数据
    df = load_raw_data()
    # 获取数据集基本信息
    # df.info()
    # 选择部分信息存入字典
    info = {
        "num_samples": df.shape[0],
        "num_features": df.shape[1],
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict()
    }
    return info


def data_loader() -> pd.DataFrame:
    """
    从数据库中获取数据或者是从csv中读取数据
    获取数据基本结构
    :return:返回原始数据
    """
    save_to_mysql(load_raw_data(), config.TABLE_NAME, get_mysql_engine())
    df = load_raw_data()
    # df = load_from_mysql(get_mysql_engine(), config.TABLE_NAME)
    print(f'基本数据信息')
    info = get_basic_info()
    pprint(info)
    return df


if __name__ == '__main__':
    # df = load_raw_data()
    data_loader()
    print(f'{time.time() - config.START:.2f}s')
