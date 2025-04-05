# 使用AAPL数据训练，MSFT微调，并预测
import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 数据获取与预处理函数
def get_stock_data(ticker, start, end):
    stock = yf.download(ticker, start=start, end=end)
    return stock['Close'].values.reshape(-1, 1)

def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# 获取AAPL和MSFT数据
aapl_data = get_stock_data('AAPL', '2020-01-01', '2025-04-05')
msft_data = get_stock_data('MSFT', '2020-01-01', '2025-04-05')

# 数据归一化
scaler_aapl = MinMaxScaler(feature_range=(0, 1))
scaler_msft = MinMaxScaler(feature_range=(0, 1))
aapl_scaled = scaler_aapl.fit_transform(aapl_data)
msft_scaled = scaler_msft.fit_transform(msft_data)

# 创建数据集
time_step = 60
X_aapl, y_aapl = create_dataset(aapl_scaled, time_step)
X_msft, y_msft = create_dataset(msft_scaled, time_step)

# 转换为PyTorch张量
X_aapl = torch.FloatTensor(X_aapl).reshape(-1, time_step, 1)
y_aapl = torch.FloatTensor(y_aapl).reshape(-1, 1)
X_msft = torch.FloatTensor(X_msft).reshape(-1, time_step, 1)
y_msft = torch.FloatTensor(y_msft).reshape(-1, 1)

# 定义LSTM模型
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 设备设置
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = StockLSTM().to(device)

# 预训练（用AAPL数据）
X_aapl, y_aapl = X_aapl.to(device), y_aapl.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

training_epoch=500
fine_tuning_epoch=50
for epoch in range(training_epoch):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_aapl)
    loss = criterion(outputs, y_aapl)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Pretrain Epoch [{epoch+1}/{training_epoch}], Loss: {loss.item():.4f}')

# 微调（用MSFT数据）
X_msft_train, y_msft_train = X_msft[:int(len(X_msft)*0.8)].to(device), y_msft[:int(len(y_msft)*0.8)].to(device)
X_msft_test, y_msft_test = X_msft[int(len(X_msft)*0.8):].to(device), y_msft[int(len(y_msft)*0.8):].to(device)
for epoch in range(fine_tuning_epoch):  # 少量epoch微调
    model.train()
    optimizer.zero_grad()
    outputs = model(X_msft_train)
    loss = criterion(outputs, y_msft_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 5 == 0:
        print(f'Finetune Epoch [{epoch+1}/{fine_tuning_epoch}], Loss: {loss.item():.4f}')

# 预测MSFT
model.eval()
with torch.no_grad():
    predictions = model(X_msft_test).cpu().numpy()
    predictions = scaler_msft.inverse_transform(predictions)
    y_msft_actual = scaler_msft.inverse_transform(y_msft_test.cpu().numpy())

# 可视化
msft_dates = yf.download('MSFT', start='2020-01-01', end='2025-04-05').index[time_step + 1 + int(len(X_msft)*0.8):]
plt.plot(msft_dates, predictions, label='Predicted')
plt.plot(msft_dates, y_msft_actual, label='Actual')
plt.legend()
plt.show()