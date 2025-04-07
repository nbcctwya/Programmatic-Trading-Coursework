import yfinance as yf
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


# 1. 获取股票数据
def get_stock_data(ticker, start_date, end_date):
    stock = yf.download(ticker, start=start_date, end=end_date)
    return stock[['Close']]


# 2. 特征工程：计算每日收益率并生成目标变量
def prepare_data(df, lookback=5):
    # 计算日收益率
    df['Return'] = df['Close'].pct_change()

    # 创建特征：前lookback天的收益率
    for i in range(1, lookback + 1):
        df[f'Return_Lag_{i}'] = df['Return'].shift(i)

    # 创建目标变量：下一交易日的涨跌（1为涨，0为跌）
    df['Target'] = (df['Return'].shift(-1) > 0).astype(int)

    # 删除NaN值
    df = df.dropna()

    # 特征和目标
    X = df[[f'Return_Lag_{i}' for i in range(1, lookback + 1)]].values
    y = df['Target'].values
    return X, y, df


# 3. 训练和预测
def train_svm(X, y):
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 分割训练和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=False)

    # 初始化SVM模型（使用RBF核）
    model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

    # 训练模型
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))

    return model, scaler, X_test, y_test, y_pred


# 4. 可视化预测结果
def plot_predictions(df, y_test, y_pred, test_indices):
    plt.figure(figsize=(10, 6))
    plt.plot(test_indices[-20:], y_test[-20:], label='Actual (0=Down, 1=Up)', marker='o')
    plt.plot(test_indices[-20:], y_pred[-20:], label='Predicted', marker='x')
    plt.title('SVM Stock Price Direction Prediction')
    plt.xlabel('Time')
    plt.ylabel('Direction (0=Down, 1=Up)')
    plt.legend()
    plt.show()


# 主程序
if __name__ == "__main__":
    # 参数设置
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2025-04-07"
    lookback = 5  # 使用前5天的收益率作为特征

    # 获取数据
    stock_data = get_stock_data(ticker, start_date, end_date)

    # 准备数据
    X, y, df = prepare_data(stock_data, lookback)

    # 训练SVM并预测
    model, scaler, X_test, y_test, y_pred = train_svm(X, y)

    # 获取测试集对应的时间索引
    test_indices = df.index[-len(y_test):]

    # 可视化
    plot_predictions(df, y_test, y_pred, test_indices)