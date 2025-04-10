# split_stock_data.py

import os
import pandas as pd

csv_folder = '../data/csv'
output_train_folder = '../data/train'
output_test_folder = '../data/test'

os.makedirs(output_train_folder, exist_ok=True)
os.makedirs(output_test_folder, exist_ok=True)

for filename in os.listdir(csv_folder):
    if filename.endswith('.csv'):
        filepath = os.path.join(csv_folder, filename)
        df = pd.read_csv(filepath)

        # 按日期排序，避免未来数据泄露
        df = df.sort_values('date')
        split_idx = int(len(df) * 0.8)

        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        train_df.to_csv(os.path.join(output_train_folder, filename), index=False)
        test_df.to_csv(os.path.join(output_test_folder, filename), index=False)

        print(f"Split {filename}: Train={len(train_df)} rows, Test={len(test_df)} rows")

print("✅ 所有文件已按 80/20 拆分完成。")
