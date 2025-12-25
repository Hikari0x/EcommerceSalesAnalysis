import time
from config import START
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def train_model(
        df: pd.DataFrame,
        target_col: str,
        test_size: float = 0.2,
        random_state: int = 123,
        model_type: str='rf'
):
    """
    模型训练模块
    :param df: 特征工程后的数据
    :param target_col: 目标列
    :return: 训练好的模型 + 测试集
    """
    # 1. 拆分特征和标签
    x = df.drop(columns=[target_col])
    y = df[target_col]
    # 2. 划分训练集 / 测试集
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    # 3. 构建模型
    # model = RandomForestClassifier(
    #     n_estimators=100,
    #     random_state=random_state,
    #     n_jobs=-1
    # )
    if model_type == 'rf':
        # 随机森林
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1
        )
    elif model_type == 'lr':
        # 逻辑回归
        model = LogisticRegression(
            max_iter=1000,
            random_state=random_state
        )
    else:
        raise ValueError(f'不支持的模型类型: {model_type}')
    # 4. 训练模型
    model.fit(x_train, y_train)
    return model, x_test, y_test


if __name__ == '__main__':
    print(f'{time.time() - START:.2f}s')
