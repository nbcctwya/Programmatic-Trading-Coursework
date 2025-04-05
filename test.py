import torch
import yfinance as yf
print("PyTorch Version:", torch.__version__)
print("MPS Available:", torch.backends.mps.is_available())
data = yf.download('AAPL', start='2025-01-01', end='2025-04-05')
print(data)