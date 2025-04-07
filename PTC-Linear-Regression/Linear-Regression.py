import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# 1. 获取股票数据
def get_stock_data(ticker, start_date, end_date):
    stock = yf.download(ticker, start=start_date, end=end_date)
    return stock['Close'].values


# 2. 数据预处理
def prepare_data(data, lookback=5):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))

    X, y = [], []
    for i in range(lookback, len(data_scaled)):
        X.append(data_scaled[i - lookback:i, 0])
        y.append(data_scaled[i, 0])
    return np.array(X), np.array(y), scaler


# 3. 定义线性回归模型
class LinearRegression(nn.Module):
    def __init__(self, input_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)


# 4. 训练模型（修复设备问题）
def train_model(model, X_train, y_train, device, epochs=100, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 将数据移到指定设备
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1).to(device)

    for epoch in range(epochs):
        # 前向传播
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')


# 5. 预测函数（修复设备问题）
def predict(model, X_test, scaler, device):
    model.eval()
    with torch.no_grad():
        inputs = torch.FloatTensor(X_test).to(device)  # 数据移到MPS
        predictions = model(inputs).cpu().numpy()  # 结果移回CPU用于后续处理
    return scaler.inverse_transform(predictions)


# 主程序
if __name__ == "__main__":
    # 参数设置
    ticker = "AAPL"
    start_date = "2023-01-01"
    end_date = "2025-04-07"
    lookback = 5

    # 获取数据
    stock_data = get_stock_data(ticker, start_date, end_date)

    # 准备数据
    X, y, scaler = prepare_data(stock_data, lookback)

    # 分割训练和测试集
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 设置设备
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化模型
    model = LinearRegression(input_size=lookback).to(device)
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)

    # 训练
    train_model(model, X_train, y_train, device, epochs=1000, lr=0.01)

    # 预测
    predictions = predict(model, X_test, scaler, device)
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_real, label="Actual Price")
    plt.plot(predictions, label="Predicted Price")
    plt.title(f"{ticker} Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()