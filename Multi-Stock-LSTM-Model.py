import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# 1. 数据准备
stocks = ['AAPL', 'MSFT', 'GOOGL']  # 示例股票
start_date = '2020-01-01'
end_date = '2025-04-05'

# 获取多股票数据
data_dict = {}
scalers = {}
for stock in stocks:
    df = yf.download(stock, start=start_date, end=end_date)
    data_dict[stock] = df['Close'].values.reshape(-1, 1)
    scalers[stock] = MinMaxScaler(feature_range=(0, 1))
    data_dict[stock] = scalers[stock].fit_transform(data_dict[stock])

# 创建时间序列数据集
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X_all, y_all, stock_ids = [], [], []
for idx, stock in enumerate(stocks):
    X, y = create_dataset(data_dict[stock], time_step)
    X_all.append(X)
    y_all.append(y)
    stock_ids.append(np.full(len(X), idx))  # 为每只股票分配ID

# 合并数据
X_all = np.concatenate(X_all, axis=0)
y_all = np.concatenate(y_all, axis=0)
stock_ids = np.concatenate(stock_ids, axis=0)

# 转换为PyTorch张量
X_all = torch.FloatTensor(X_all).reshape(-1, time_step, 1)
y_all = torch.FloatTensor(y_all).reshape(-1, 1)
stock_ids = torch.LongTensor(stock_ids)

# 创建自定义数据集
class StockDataset(Dataset):
    def __init__(self, X, y, stock_ids):
        self.X = X
        self.y = y
        self.stock_ids = stock_ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.stock_ids[idx]

dataset = StockDataset(X_all, y_all, stock_ids)
train_size = int(len(dataset) * 0.8)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 2. 定义多股票模型
class MultiStockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, num_stocks=len(stocks), embedding_dim=10):
        super(MultiStockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 股票嵌入层
        self.embedding = nn.Embedding(num_stocks, embedding_dim)
        # LSTM层
        self.lstm = nn.LSTM(input_size + embedding_dim, hidden_size, num_layers, batch_first=True)
        # 输出层
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, stock_ids):
        # 获取股票嵌入
        stock_emb = self.embedding(stock_ids)  # [batch_size, embedding_dim]
        stock_emb = stock_emb.unsqueeze(1).repeat(1, x.size(1), 1)  # [batch_size, time_step, embedding_dim]
        # 将嵌入与输入拼接
        x = torch.cat((x, stock_emb), dim=2)  # [batch_size, time_step, input_size + embedding_dim]
        # LSTM前向传播
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        # 输出最后一时间步的预测
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
    for X_batch, y_batch, stock_ids_batch in train_loader:
        X_batch, y_batch, stock_ids_batch = X_batch.to(device), y_batch.to(device), stock_ids_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch, stock_ids_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')

# 4. 测试与预测
model.eval()
predictions_dict = {stock: [] for stock in stocks}
actual_dict = {stock: [] for stock in stocks}
with torch.no_grad():
    for X_batch, y_batch, stock_ids_batch in test_loader:
        X_batch, stock_ids_batch = X_batch.to(device), stock_ids_batch.to(device)
        preds = model(X_batch, stock_ids_batch).cpu().numpy()
        y_batch = y_batch.numpy()
        for i, stock_idx in enumerate(stock_ids_batch.cpu().numpy()):
            stock = stocks[stock_idx]
            predictions_dict[stock].append(preds[i])
            actual_dict[stock].append(y_batch[i])

# 反归一化并可视化
for stock in stocks:
    preds = scalers[stock].inverse_transform(np.array(predictions_dict[stock]))
    actual = scalers[stock].inverse_transform(np.array(actual_dict[stock]))
    plt.figure()
    plt.plot(preds, label='Predicted')
    plt.plot(actual, label='Actual')
    plt.title(f'{stock} Stock Price Prediction')
    plt.legend()
    plt.show()