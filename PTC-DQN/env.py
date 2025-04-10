# 第一步：env.py
import gym
import numpy as np

class StockTradingEnv(gym.Env):
    def __init__(self, df, initial_cash=10000):
        super(StockTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_cash = initial_cash
        self.action_space = gym.spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.stock_owned = 0
        self.total_asset = self.cash
        return self._get_observation()

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        obs = np.array([
            row['open'], row['high'], row['low'], row['close'],
            self.stock_owned, self.cash
        ], dtype=np.float32)
        return obs

    def step(self, action):
        done = False
        row = self.df.iloc[self.current_step]
        price = row['close']

        # 执行操作
        if action == 1:  # Buy
            if self.cash > price:
                self.stock_owned += 1
                self.cash -= price
        elif action == 2:  # Sell
            if self.stock_owned > 0:
                self.stock_owned -= 1
                self.cash += price

        self.total_asset = self.cash + self.stock_owned * price

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True

        reward = self.total_asset - self.initial_cash  # 简单收益
        next_state = self._get_observation()

        return next_state, reward, done, {}