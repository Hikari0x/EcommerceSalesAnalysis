import time

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report

from config import START
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from joblib import dump


def _build_model(model_type: str, random_state: int):
    """
    构建模型（内部函数）
    :param model_type: 模型类型
    :param random_state: 随机种子
    """
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1
        )
    elif model_type == "lr":
        return LogisticRegression(
            max_iter=1000,
            random_state=random_state
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


def _evaluate(model, x_test, y_test) -> dict:
    """
    模型评估（内部函数）
    :param model: 模型
    :param x_test: 测试集特征
    :param y_test: 测试集标签
    :return:
        metrics: 评估指标
    """
    y_pred = model.predict(x_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        )
    }


def train_and_evaluate(
        df: pd.DataFrame,
        target_col: str,
        test_size: float = 0.2,
        random_state: int = 123,
        model_type: str = 'rf'
):
    """
    模型训练模块和评估一体化
    :param df: 特征工程后的数据
    :param target_col: 目标列
    :return:
        model: 训练好的模型
        metrics: 评估指标
    """
    # 1. 拆分特征和标签
    x = df.drop(columns=[target_col])
    y = df[target_col]
    # 2. 划分训练集 / 测试集
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    # 3. 构建模型
    model = _build_model(model_type, random_state)
    # 4. 训练模型
    model.fit(x_train, y_train)
    # 保存模型
    dump(model, f'../model/{model_type}.joblib')
    # 5. 评估模型
    metrics = _evaluate(model, x_test, y_test)

    return model, metrics


if __name__ == '__main__':
    print(f'{time.time() - START:.2f}s')
