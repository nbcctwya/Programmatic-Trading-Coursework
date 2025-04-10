# 第五步：main.py
from env import StockTradingEnv
from agent import DQNAgent
from utils import load_data
import numpy as np

train_path = './data/train/000001.csv'
test_path = './data/test/000001.csv'

df_train = load_data(train_path)
df_test = load_data(test_path)

env = StockTradingEnv(df_train)
agent = DQNAgent(state_dim=6, action_dim=3)

EPISODES = 50
for e in range(EPISODES):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.replay()
        state = next_state
        total_reward += reward
    print(f"Episode {e+1}/{EPISODES} - Total Asset: {env.total_asset:.2f}, Reward: {total_reward:.2f}")
