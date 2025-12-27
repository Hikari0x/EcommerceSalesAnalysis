import time
from config import START
import torch
import torch.nn as nn


class DLModel(nn.Module):
    """
        最简单的全连接神经网络（不含 Embedding）
        先跑通流程用
        """

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def train_dnn(model, x, y, epochs=10, lr=0.001):
    """
    训练简单 DNN 模型
    参数：
    X : 特征（numpy）
    y : 标签
    :param model:模型
    :param x:
    :param epochs:训练轮数
    :param lr:学习率
    :return:
    """
    device = "cpu"
    model.to(device)
    X_tensor = torch.tensor(x.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X_tensor)
        loss = criterion(preds, y_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")


if __name__ == '__main__':
    print(f'{time.time() - START:.2f}s')
