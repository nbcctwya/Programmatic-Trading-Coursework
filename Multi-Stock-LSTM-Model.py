import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta

# 1. 数据准备
stocks = ['AAPL', 'MSFT', 'GOOGL']
start_date = '2020-01-01'
end_date = '2025-04-05'
test_start_date = '2024-04-05'  # 测试集从最近一年开始

# 获取多股票数据并保留日期
data_dict = {}
dates_dict = {}
scalers = {}
for stock in stocks:
    df = yf.download(stock, start=start_date, end=end_date)
    data_dict[stock] = df['Close'].values.reshape(-1, 1)
    dates_dict[stock] = df.index
    scalers[stock] = MinMaxScaler(feature_range=(0, 1))
    data_dict[stock] = scalers[stock].fit_transform(data_dict[stock])


# 创建时间序列数据集
def create_dataset(data, dates, time_step=60, test_start_date=None):
    X, y, date_indices = [], [], []
    test_start_idx = len(dates)  # 默认全部数据
    if test_start_date:
        test_start = datetime.strptime(test_start_date, '%Y-%m-%d')
        test_start_idx = next(i for i, d in enumerate(dates) if d >= test_start) - time_step - 1
        if test_start_idx < 0:
            test_start_idx = 0

    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
        date_indices.append(i + time_step + 1)  # 预测的日期索引

    X, y, date_indices = np.array(X), np.array(y), np.array(date_indices)
    # 分割训练和测试集
    X_train, y_train, date_train = X[:test_start_idx], y[:test_start_idx], date_indices[:test_start_idx]
    X_test, y_test, date_test = X[test_start_idx:], y[test_start_idx:], date_indices[test_start_idx:]
    return X_train, y_train, date_train, X_test, y_test, date_test


time_step = 60
X_train_all, y_train_all, date_train_all, X_test_all, y_test_all, date_test_all, stock_ids_train, stock_ids_test = [], [], [], [], [], [], [], []
for idx, stock in enumerate(stocks):
    X_train, y_train, date_train, X_test, y_test, date_test = create_dataset(
        data_dict[stock], dates_dict[stock], time_step, test_start_date
    )
    X_train_all.append(X_train)
    y_train_all.append(y_train)
    date_train_all.append(date_train)
    X_test_all.append(X_test)
    y_test_all.append(y_test)
    date_test_all.append(date_test)
    stock_ids_train.append(np.full(len(X_train), idx))
    stock_ids_test.append(np.full(len(X_test), idx))

# 合并数据
X_train_all = np.concatenate(X_train_all, axis=0)
y_train_all = np.concatenate(y_train_all, axis=0)
date_train_all = np.concatenate(date_train_all, axis=0)
stock_ids_train = np.concatenate(stock_ids_train, axis=0)
X_test_all = np.concatenate(X_test_all, axis=0)
y_test_all = np.concatenate(y_test_all, axis=0)
date_test_all = np.concatenate(date_test_all, axis=0)
stock_ids_test = np.concatenate(stock_ids_test, axis=0)

# 转换为PyTorch张量
X_train_all = torch.FloatTensor(X_train_all).reshape(-1, time_step, 1)
y_train_all = torch.FloatTensor(y_train_all).reshape(-1, 1)
stock_ids_train = torch.LongTensor(stock_ids_train)
X_test_all = torch.FloatTensor(X_test_all).reshape(-1, time_step, 1)
y_test_all = torch.FloatTensor(y_test_all).reshape(-1, 1)
stock_ids_test = torch.LongTensor(stock_ids_test)


# 创建自定义数据集
class StockDataset(Dataset):
    def __init__(self, X, y, stock_ids, date_indices):
        self.X = X
        self.y = y
        self.stock_ids = stock_ids
        self.date_indices = date_indices

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.stock_ids[idx], self.date_indices[idx]


train_dataset = StockDataset(X_train_all, y_train_all, stock_ids_train, date_train_all)
test_dataset = StockDataset(X_test_all, y_test_all, stock_ids_test, date_test_all)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 2. 定义多股票模型（保持不变）
class MultiStockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, num_stocks=len(stocks), embedding_dim=10):
        super(MultiStockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_stocks, embedding_dim)
        self.lstm = nn.LSTM(input_size + embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, stock_ids):
        stock_emb = self.embedding(stock_ids)
        stock_emb = stock_emb.unsqueeze(1).repeat(1, x.size(1), 1)
        x = torch.cat((x, stock_emb), dim=2)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# 3. 训练模型
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = MultiStockLSTM().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch, stock_ids_batch, _ in train_loader:
        X_batch, y_batch, stock_ids_batch = X_batch.to(device), y_batch.to(device), stock_ids_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch, stock_ids_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}')

# 4. 测试与预测（最近一年数据）
model.eval()
predictions_dict = {stock: [] for stock in stocks}
actual_dict = {stock: [] for stock in stocks}
dates_dict_test = {stock: [] for stock in stocks}

with torch.no_grad():
    for X_batch, y_batch, stock_ids_batch, date_indices_batch in test_loader:
        X_batch, stock_ids_batch = X_batch.to(device), stock_ids_batch.to(device)
        preds = model(X_batch, stock_ids_batch).cpu().numpy()
        y_batch = y_batch.numpy()
        date_indices_batch = date_indices_batch.numpy()
        for i, (stock_idx, date_idx) in enumerate(zip(stock_ids_batch.cpu().numpy(), date_indices_batch)):
            stock = stocks[stock_idx]
            predictions_dict[stock].append(preds[i])
            actual_dict[stock].append(y_batch[i])
            dates_dict_test[stock].append(dates_dict[stock][date_idx])

# 5. 可视化（以日期为X轴）
for stock in stocks:
    preds = scalers[stock].inverse_transform(np.array(predictions_dict[stock]))
    actual = scalers[stock].inverse_transform(np.array(actual_dict[stock]))
    dates = dates_dict_test[stock]

    plt.figure(figsize=(12, 6))
    plt.plot(dates, preds, label='Predicted')
    plt.plot(dates, actual, label='Actual')
    plt.title(f'{stock} Stock Price Prediction (Last Year)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()