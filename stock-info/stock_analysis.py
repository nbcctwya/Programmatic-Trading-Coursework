import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置路径
data_dir = '../data/csv'
results = []

# 遍历所有CSV文件
for file in os.listdir(data_dir):
    if file.endswith('.csv'):
        code = file.replace('.csv', '')
        df = pd.read_csv(os.path.join(data_dir, file))

        # 日期排序
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # 计算每日收益率
        df['return'] = df['close'].pct_change()
        df = df.dropna()

        if len(df) < 10:
            continue  # 太少的数据不统计

        # 累计收益率
        total_return = df['close'].iloc[-1] / df['close'].iloc[0] - 1

        # 年化收益率
        n_days = len(df)
        annual_return = (1 + total_return) ** (252 / n_days) - 1

        # 年化波动率
        annual_volatility = df['return'].std() * np.sqrt(252)

        # 夏普比率（无风险利率假设为0）
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else np.nan

        # 最大回撤
        cumulative = (1 + df['return']).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()

        results.append({
            'stock': code,
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        })

# 转换为DataFrame
result_df = pd.DataFrame(results)

# 保存结果（可选）
result_df.to_csv('./stock_stats_summary.csv', index=False)

# 可视化：波动率分布
plt.figure(figsize=(10, 6))
plt.hist(result_df['annual_volatility'].dropna(), bins=20, edgecolor='black', alpha=0.7)
plt.title('年化波动率分布图')
plt.xlabel('年化波动率')
plt.ylabel('股票数量')
plt.grid(True)
plt.tight_layout()
plt.show()