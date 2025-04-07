import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 定义参数
TICKER = 'MSFT'  # 股票代码
START_DATE = '2018-01-01'
END_DATE = '2023-12-31'
SPLIT_RATIO = 0.8  # 训练集比例
SEQUENCE_LENGTH = 60  # 使用过去60天数据进行预测
HIDDEN_SIZE = 64  # LSTM隐藏层大小
NUM_LAYERS = 2  # LSTM层数
DROPOUT = 0.2  # Dropout比例
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 500
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')  # 使用M3 Pro上的MPS加速

print(f"使用设备: {DEVICE}")


# 下载股票数据
def download_stock_data(ticker, start_date, end_date):
    print(f"正在下载 {ticker} 的历史数据...")
    data = yf.download(ticker, start=start_date, end=end_date)
    print(f"已下载 {len(data)} 条数据")
    return data


# 准备数据集
class StockDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length, :]
        y = self.data[idx + self.sequence_length, 0]  # 预测收盘价
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 全连接层
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播LSTM
        out, _ = self.lstm(x, (h0, c0))

        # 只取序列的最后一个输出
        out = self.fc(out[:, -1, :])
        return out.squeeze()


# 训练模型
def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

    return train_loss / len(train_loader.dataset)


# 评估模型
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    predictions = []
    actuals = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item() * inputs.size(0)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())

    return val_loss / len(test_loader.dataset), predictions, actuals


# 主函数
def main():
    # 下载数据
    stock_data = download_stock_data(TICKER, START_DATE, END_DATE)

    # 只使用收盘价
    close_prices = stock_data['Close'].values.reshape(-1, 1)

    # 数据标准化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # 分割训练集和测试集
    train_size = int(len(scaled_data) * SPLIT_RATIO)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - SEQUENCE_LENGTH:]  # 保留一些重叠，确保测试集有足够的序列

    # 创建数据集
    train_dataset = StockDataset(train_data, SEQUENCE_LENGTH)
    test_dataset = StockDataset(test_data, SEQUENCE_LENGTH)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 初始化模型
    model = LSTMModel(
        input_size=1,  # 只使用收盘价
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 训练模型
    print("开始训练模型...")
    # train_losses = []
    # val_losses = []

    for epoch in range(EPOCHS):
        # 训练
        train_loss = train_model(model, train_loader, optimizer, criterion, DEVICE)

        # train_losses.append(train_loss)
        # val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            # 评估
            val_loss, _, _ = evaluate_model(model, test_loader, criterion, DEVICE)
            print(f"Epoch {epoch + 1}/{EPOCHS}, 训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}")

    # 最终评估
    _, test_predictions, test_actuals = evaluate_model(model, test_loader, criterion, DEVICE)

    # 反标准化数据
    test_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1)).flatten()
    test_actuals = scaler.inverse_transform(np.array(test_actuals).reshape(-1, 1)).flatten()

    # 计算指标
    mse = mean_squared_error(test_actuals, test_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_actuals, test_predictions)

    print(f"\n测试集结果:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")

    # 绘制结果
    plt.figure(figsize=(12, 6))


    # 预测结果
    plt.subplot(1, 2, 2)
    plt.plot(test_actuals, label='actual')
    plt.plot(test_predictions, label='predict')
    plt.xlabel('time')
    plt.ylabel('price')
    plt.title(f'{TICKER} stock price')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 保存模型
    torch.save(model.state_dict(), f'lstm_{TICKER}_model.pth')
    print(f"模型已保存为 lstm_{TICKER}_model.pth")

    # 预测未来一天
    predict_future(model, scaled_data, scaler)


def predict_future(model, scaled_data, scaler):
    # 准备最后一个序列作为输入
    last_sequence = scaled_data[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, 1)
    last_sequence_tensor = torch.tensor(last_sequence, dtype=torch.float32).to(DEVICE)

    # 预测
    model.eval()
    with torch.no_grad():
        prediction = model(last_sequence_tensor).cpu().numpy()
        prediction = scaler.inverse_transform(prediction.reshape(-1, 1))

    # 获取最新日期
    last_date = pd.to_datetime(END_DATE) + pd.Timedelta(days=1)

    print(f"\n预测 {last_date.strftime('%Y-%m-%d')} 的收盘价: ${prediction[0, 0]:.2f}")


if __name__ == "__main__":
    main()