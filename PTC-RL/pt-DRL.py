import numpy as np
import pandas as pd
import yfinance as yf
import gymnasium
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env


# 获取股票数据
def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data


# 自定义环境
class StockTradingEnv(gymnasium.Env):
    def __init__(self, data):
        super(StockTradingEnv, self).__init__()
        self.data = data.reset_index()
        self.current_step = 0
        self.initial_cash = 10000
        self.cash = self.initial_cash
        self.holdings = 0
        self.action_space = spaces.Discrete(3)  # 0=Buy, 1=Sell, 2=Hold
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32)
        self.seed_value = None  # 用于存储随机种子

    def seed(self, seed=None):
        """设置随机种子"""
        self.seed_value = seed
        np.random.seed(seed)
        return [seed]

    def _get_state(self):
        return np.array([
            self.data['Close'].iloc[self.current_step],
            self.data['Volume'].iloc[self.current_step]
        ], dtype=np.float32)

    def step(self, action):
        current_price = self.data['Close'].iloc[self.current_step]
        if action == 0:  # Buy
            shares_to_buy = self.cash // current_price
            self.holdings += shares_to_buy
            self.cash -= shares_to_buy * current_price
        elif action == 1 and self.holdings > 0:  # Sell
            self.cash += self.holdings * current_price
            self.holdings = 0

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        total_value = self.cash + self.holdings * current_price
        reward = total_value - self.initial_cash
        return self._get_state(), reward, done, {}

    def reset(self, seed=None):
        """重置环境，支持种子参数"""
        if seed is not None:
            self.seed(seed)  # 如果提供了种子，则设置
        self.current_step = 0
        self.cash = self.initial_cash
        self.holdings = 0
        return self._get_state()

    def render(self, mode='human'):
        total_value = self.cash + self.holdings * self.data['Close'].iloc[self.current_step]
        print(
            f"Step: {self.current_step}, Cash: {self.cash:.2f}, Holdings: {self.holdings}, Total Value: {total_value:.2f}")


# 主程序
if __name__ == "__main__":
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2025-04-07"
    stock_data = fetch_stock_data(ticker, start_date, end_date)

    env = StockTradingEnv(stock_data)
    check_env(env)  # 检查环境

    model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.001)
    model.learn(total_timesteps=10000)

    obs = env.reset()
    total_rewards = 0
    for _ in range(len(stock_data)):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_rewards += reward
        env.render()
        if done:
            break

    print(f"Total Rewards: {total_rewards:.2f}")