# 第四步：utils.py
import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df = df[['open', 'high', 'low', 'close']]
    return df