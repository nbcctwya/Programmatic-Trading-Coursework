import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def calculate_metrics(df):
    df = df.sort_values('date')
    df['return'] = df['close'].pct_change()
    df.dropna(inplace=True)

    df['cumulative_return'] = (1 + df['return']).cumprod()

    cumulative_return = df['cumulative_return'].iloc[-1] - 1
    avg_daily_return = df['return'].mean()
    std_daily_return = df['return'].std()
    annualized_return = avg_daily_return * 252
    annualized_volatility = std_daily_return * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else np.nan

    # 最大回撤
    cumulative = df['cumulative_return']
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    return {
        'cumulative_return': cumulative_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'returns_series': df['return']
    }

def build_gui(data_folder='../data/csv'):
    root = tk.Tk()
    root.title("📈 股票数据分析工具")
    root.geometry("720x600")

    files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

    label = tk.Label(root, text="选择股票文件：", font=('Arial', 12))
    label.pack(pady=10)

    combo = ttk.Combobox(root, values=files, font=('Arial', 11))
    combo.pack(pady=5)

    text = tk.Text(root, height=10, width=80, font=('Courier', 10))
    text.pack(pady=10)

    canvas_widget = None  # 用于后续更新图像

    def on_select():
        nonlocal canvas_widget

        file = combo.get()
        filepath = os.path.join(data_folder, file)
        stock_id = file.replace('.csv', '')
        df = pd.read_csv(filepath)

        if 'date' not in df.columns or 'close' not in df.columns:
            text.delete('1.0', tk.END)
            text.insert(tk.END, f"{file} 缺少必要列（需要至少包含 'date' 和 'close'）。")
            return

        metrics = calculate_metrics(df)
        display_text = f"""
股票编号：{stock_id}
累计收益率：{metrics['cumulative_return']:.2%}
年化收益率：{metrics['annualized_return']:.2%}
年化波动率：{metrics['annualized_volatility']:.2%}
夏普比率：{metrics['sharpe_ratio']:.2f}
最大回撤：{metrics['max_drawdown']:.2%}
        """
        text.delete('1.0', tk.END)
        text.insert(tk.END, display_text)

        # 清除旧图像（若存在）
        if canvas_widget:
            canvas_widget.get_tk_widget().pack_forget()

        # 生成波动率分布图
        fig, ax = plt.subplots(figsize=(6, 3.5), dpi=100)
        sns.histplot(metrics['returns_series'], bins=50, kde=True, ax=ax, color='steelblue')
        # ax.set_title(f"波动率分布图 - {stock_id}", fontsize=12)
        ax.set_title(f"Volatility Distribution Chart - {stock_id}", fontsize=12)
        # ax.set_xlabel("日收益率")
        # ax.set_ylabel("频率")
        ax.set_xlabel("Daily Rate of Return")
        ax.set_ylabel("frequency")
        plt.tight_layout()

        canvas_widget = FigureCanvasTkAgg(fig, master=root)
        canvas_widget.draw()
        canvas_widget.get_tk_widget().pack(pady=10)

    # btn = tk.Button(root, text="开始分析", command=on_select, font=('Arial', 12), bg="#4CAF50", fg="white")
    btn = tk.Button(root, text="Start analysis", command=on_select, font=('Arial', 12), bg="#4CAF50", fg="white")
    btn.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    build_gui()
