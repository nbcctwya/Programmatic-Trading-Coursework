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
EPOCHS = 5000
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


# 计算交易信号和收益率
def calculate_returns(actual_prices, predicted_prices, dates):
    df = pd.DataFrame({
        'Date': dates,
        'Actual': actual_prices,
        'Predicted': predicted_prices
    })

    # 计算价格变化的预测
    df['Actual_Change'] = df['Actual'].pct_change()
    df['Predicted_Change'] = df['Predicted'].pct_change()

    # 生成交易信号: 1表示买入, -1表示卖出, 0表示持有现金
    df['Signal'] = np.where(df['Predicted_Change'] > 0, 1, -1)

    # 根据前一天的信号计算收益率
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Actual_Change']

    # 按信号计算累积收益率
    df['Cum_Market_Return'] = (1 + df['Actual_Change']).cumprod() - 1
    df['Cum_Strategy_Return'] = (1 + df['Strategy_Return']).cumprod() - 1

    # 计算绩效指标
    strategy_return = df['Cum_Strategy_Return'].iloc[-1]
    market_return = df['Cum_Market_Return'].iloc[-1]

    # 年化收益率 (假设250个交易日/年)
    n_years = len(df) / 250
    strategy_annual_return = (1 + strategy_return) ** (1 / n_years) - 1
    market_annual_return = (1 + market_return) ** (1 / n_years) - 1

    # 计算夏普比率 (假设无风险利率为0)
    daily_returns = df['Strategy_Return'].dropna()
    sharpe_ratio = np.sqrt(250) * daily_returns.mean() / daily_returns.std()

    # 计算最大回撤
    cum_returns = (1 + df['Strategy_Return'].fillna(0)).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max) - 1
    max_drawdown = drawdown.min()

    # 计算胜率
    win_rate = (df['Strategy_Return'] > 0).sum() / (df['Strategy_Return'] != 0).sum()

    # 统计指标
    metrics = {
        "总收益率": f"{strategy_return:.2%}",
        "年化收益率": f"{strategy_annual_return:.2%}",
        "市场收益率": f"{market_return:.2%}",
        "市场年化收益率": f"{market_annual_return:.2%}",
        "超额收益": f"{strategy_return - market_return:.2%}",
        "夏普比率": f"{sharpe_ratio:.2f}",
        "最大回撤": f"{max_drawdown:.2%}",
        "胜率": f"{win_rate:.2%}"
    }

    return df, metrics


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
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        # 训练
        train_loss = train_model(model, train_loader, optimizer, criterion, DEVICE)

        # 评估
        val_loss, _, _ = evaluate_model(model, test_loader, criterion, DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}, 训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}")

    # 最终评估
    _, test_predictions, test_actuals = evaluate_model(model, test_loader, criterion, DEVICE)

    # 反标准化数据
    test_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1)).flatten()
    test_actuals = scaler.inverse_transform(np.array(test_actuals).reshape(-1, 1)).flatten()

    # 获取测试数据的日期
    test_dates = stock_data.index[train_size:train_size + len(test_actuals)]

    # 计算收益率
    print("\n计算交易策略收益率...")
    returns_df, metrics = calculate_returns(test_actuals, test_predictions, test_dates)

    # 打印绩效指标
    print("\n========== 策略绩效指标 ==========")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    # 绘制图表
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))

    # 训练/验证损失
    axes[0, 0].plot(train_losses, label='train loss')
    axes[0, 0].plot(val_losses, label='eval loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('loss')
    axes[0, 0].set_title('train and eval loss')
    axes[0, 0].legend()

    # 价格预测
    axes[0, 1].plot(test_dates, test_actuals, label='actual price')
    axes[0, 1].plot(test_dates, test_predictions, label='predict price')
    axes[0, 1].set_xlabel('date')
    axes[0, 1].set_ylabel('price')
    axes[0, 1].set_title(f'{TICKER} stock prediction')
    axes[0, 1].legend()

    # 交易信号
    buy_signals = returns_df[returns_df['Signal'] == 1].index
    sell_signals = returns_df[returns_df['Signal'] == -1].index

    axes[1, 0].plot(returns_df.index, returns_df['Actual'], label='stock price')
    axes[1, 0].scatter(buy_signals, returns_df.loc[buy_signals, 'Actual'],
                       color='green', marker='^', label='buy signal')
    axes[1, 0].scatter(sell_signals, returns_df.loc[sell_signals, 'Actual'],
                       color='red', marker='v', label='sell signal')
    axes[1, 0].set_xlabel('date')
    axes[1, 0].set_ylabel('price')
    axes[1, 0].set_title('trade signal visualization')
    axes[1, 0].legend()

    # 累积收益对比
    axes[1, 1].plot(returns_df.index, returns_df['Cum_Strategy_Return'], label='policy profit')
    axes[1, 1].plot(returns_df.index, returns_df['Cum_Market_Return'], label='market profit')
    axes[1, 1].set_xlabel('date')
    axes[1, 1].set_ylabel('profit accumulate')
    axes[1, 1].set_title('police vs market')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

    # 保存回测结果
    returns_df.to_csv(f'{TICKER}_strategy_returns.csv')
    print(f"策略回测结果已保存至 {TICKER}_strategy_returns.csv")

    # 保存模型
    torch.save(model.state_dict(), f'lstm_{TICKER}_model.pth')
    print(f"模型已保存为 lstm_{TICKER}_model.pth")

    # 预测未来一天并生成交易信号
    predict_future_and_signal(model, scaled_data, scaler, stock_data)


def predict_future_and_signal(model, scaled_data, scaler, stock_data):
    # 准备最后一个序列作为输入
    last_sequence = scaled_data[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, 1)
    last_sequence_tensor = torch.tensor(last_sequence, dtype=torch.float32).to(DEVICE)

    # 获取最新收盘价
    latest_close = stock_data['Close'].iloc[-1]

    # 预测
    model.eval()
    with torch.no_grad():
        prediction = model(last_sequence_tensor).cpu().numpy()
        prediction = scaler.inverse_transform(prediction.reshape(-1, 1))[0, 0]

    # 生成交易信号
    price_change = (prediction - latest_close) / latest_close
    signal = "买入" if price_change > 0 else "卖出"

    # 获取第二天日期
    next_date = pd.to_datetime(END_DATE) + pd.Timedelta(days=1)

    print(f"\n======== 未来交易信号 ========")
    print(f"预测日期: {next_date.strftime('%Y-%m-%d')}")
    print(f"最新收盘价: ${latest_close:.2f}")
    print(f"预测价格: ${prediction:.2f}")
    print(f"预测变动: {price_change:.2%}")
    print(f"交易信号: {signal}")


if __name__ == "__main__":
    main()