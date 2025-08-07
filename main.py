# Revised Backtesting Framework with Detailed Outputs

# Output Files:
# 1. output/output.csv - Strategy performance metrics per symbol
# 2. output/tradebook.csv - Detailed trade execution records
# 3. output/charts/<SYMBOL>.png - Saved chart if enabled

# output.csv columns:
# - Symbol
# - Total return %
# - Total no. of trades
# - Winning trades
# - Max drawdown
# - Sharpe ratio
# - Annual return
# - Annual volatility
# - Alpha (vs benchmark)
# - Beta (vs benchmark)
# - Return of stock in backtested timeframe

# tradebook.csv columns:
# - Symbol
# - First leg action (Buy/Sell)
# - Entry timestamp
# - Entry quantity
# - Second leg action (Sell/Buy)
# - Exit timestamp

# engine.py - run() method enhancements:
# - Collect trade entry/exit events with timestamps and quantities
# - Use PyFolio or manual calculations for returns
# - Regress strategy returns vs benchmark returns to compute Alpha and Beta
# - Save equity curve chart using cerebro.plot() and matplotlib if save_chart=True

# main.py
import os
import pandas as pd
from engine import BacktestEngine
from strategies.my_strategy import MyStrategy
from utils import load_data

# Settings
symbols = ['RELIANCE.NS', 'TCS.NS']
save_chart = True
benchmark_symbol = ["^NSEI"]

period = '2y'
interval = '1d'

benchmark = load_data(benchmark_symbol, period=period, interval=interval, is_bechmark=True)
benchmark_bt = load_data(benchmark_symbol, period=period, interval=interval)
#benchmark_bt = None
# Output containers
results = []
tradebook_all = []

# Run backtest per symbol
for symbol in symbols:
    data = load_data(symbol, period=period, interval=interval)  # Should return bt.feeds.PandasData
    engine = BacktestEngine(
        strategy=MyStrategy,
        data1=data,
        data2=benchmark_bt,
        symbol=symbol,
        benchmark=benchmark,
        save_chart=save_chart
    )
    metrics, tradebook = engine.run()
    results.append(metrics)
    tradebook_all.extend(tradebook)

# Save outputs
os.makedirs("output", exist_ok=True)
pd.DataFrame(results).to_csv("output/output.csv", index=False)
pd.DataFrame(tradebook_all).to_csv("output/tradebook.csv", index=False)

print("✅ Backtest complete. Results saved to output folder.")
