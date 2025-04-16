import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置路径
data_dir = '../data/csv'
output_dir = '../stock_return_hist'
os.makedirs(output_dir, exist_ok=True)

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
            continue  # 数据太少不统计

        # ====== 指标计算（可选）======
        total_return = df['close'].iloc[-1] / df['close'].iloc[0] - 1
        n_days = len(df)
        annual_return = (1 + total_return) ** (252 / n_days) - 1
        annual_volatility = df['return'].std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else np.nan
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

        # ====== 画日收益率分布图 ======
        plt.figure(figsize=(8, 5))
        plt.hist(df['return'], bins=50, edgecolor='black', alpha=0.7)
        plt.title(f'{code} Daily Rate of Return distribution chart')
        plt.xlabel('Daily Rate of Return')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()

        # 保存图像
        plt.savefig(os.path.join(output_dir, f'{code}_return_hist.png'))
        plt.close()

# 保存指标统计
result_df = pd.DataFrame(results)
result_df.to_csv('./stock_stats_summary.csv', index=False)

print(f"✅ 完成，已为 {len(result_df)} 支股票绘制日收益率分布图，并保存在：{output_dir}")
print(results)