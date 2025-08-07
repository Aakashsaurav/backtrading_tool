# engine.py
import backtrader as bt
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression

class TradeLogger(bt.Analyzer):
    def __init__(self):
        self.trades = []
        self._open_orders = {}
        self.final_price = None  # Will store last price seen in backtest

    def notify_order(self, order):
        if order.status == order.Completed:
            dt = bt.num2date(order.executed.dt)
            price = order.executed.price
            size = order.executed.size
            symbol = order.data._name
            self.final_price = price  # Update last seen price

            action = 'Buy' if order.isbuy() else 'Sell'

            if action == 'Buy':
                # Store open position
                self._open_orders[symbol] = {
                    'entry_time': dt,
                    'entry_price': price,
                    'size': size,
                    'action': action,
                    'commission': order.executed.comm or 0
                }

            elif action == 'Sell' and symbol in self._open_orders:
                # Complete trade
                entry = self._open_orders.pop(symbol)
                pnl = (price - entry['entry_price']) * entry['size']
                pnl_pct = (price / entry['entry_price'] - 1) * 100
                total_commission = entry['commission'] + (order.executed.comm or 0)

                self.trades.append({
                    'Symbol': symbol,
                    'First leg action': entry['action'],
                    'entry_time': entry['entry_time'],
                    'size': entry['size'],
                    'entry_price': entry['entry_price'],
                    'Second leg action': action,
                    'exit_time': dt,
                    'exit_price': price,
                    'pnl': pnl,
                    'PnL %': pnl_pct,
                    'commission': total_commission
                })

    def stop(self):
        # Log open positions (not closed by end of backtest)
        for symbol, entry in self._open_orders.items():
            # Use final available price to estimate PnL
            exit_price = self.final_price or entry['entry_price']
            pnl = (exit_price - entry['entry_price']) * entry['size']
            pnl_pct = (exit_price / entry['entry_price'] - 1) * 100

            self.trades.append({
                'Symbol': symbol,
                'First leg action': entry['action'],
                'entry_time': entry['entry_time'],
                'size': entry['size'],
                'entry_price': entry['entry_price'],
                'Second leg action': 'Open',
                'exit_time': None,
                'exit_price': None,
                'pnl': pnl,
                'PnL %': pnl_pct,
                'commission': entry['commission']
            })

    def get_analysis(self):
        return self.trades

class BacktestEngine:
    def __init__(self, strategy, data1, data2, symbol, benchmark, cash=100000, commission=0.001, save_chart=False, percents=100):
        self.strategy = strategy
        self.data = data1
        self.data2 = data2
        self.symbol = symbol
        self.benchmark = benchmark
        self.cash = cash
        self.commission = commission
        self.save_chart = save_chart

    def run(self):
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(self.cash)
        cerebro.addsizer(bt.sizers.PercentSizerInt, percents=100)
        cerebro.broker.setcommission(commission=self.commission)
        cerebro.addstrategy(self.strategy)
        cerebro.adddata(self.data)
        if self.data2 is not None:
            cerebro.adddata(self.data2)

        cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(TradeLogger, _name='trades')

        results = cerebro.run()
        strat = results[0]

        pyfolio_analysis = strat.analyzers.pyfolio.get_analysis()
        returns = pd.Series(pyfolio_analysis['returns']).dropna()
        equity = (1 + returns).cumprod()
        annual_return = equity.iloc[-1]**(252/len(equity)) - 1
        annual_volatility = returns.std() * np.sqrt(252)

        # Calculate alpha and beta
        benchmark_returns = self.benchmark.pct_change().dropna()
        common_index = returns.index.intersection(benchmark_returns.index)
        benchmark_common = benchmark_returns.loc[common_index]
        returns_common = returns.loc[common_index]

        model = LinearRegression().fit(benchmark_common.values.reshape(-1, 1), returns_common.values)
        alpha = model.intercept_ * 252
        beta = model.coef_[0]

        # Tradebook creation
        trades = strat.analyzers.trades.get_analysis()
        tradebook = []
        for t in trades:
            tradebook.append({
                'Symbol': self.symbol,
                'First leg action': 'Buy' if t['size'] > 0 else 'Sell',
                'Timestamp': t['entry_time'],
                'Quantity': abs(t['size']),
                'Order Price': t['entry_price'],
                'Second leg action': 'Sell' if t['size'] > 0 else 'Buy',
                'Exit timestamp': t['exit_time'],
                'Exit Price': t['exit_price']
            })

        # Metrics
        metrics = {
            'Symbol': self.symbol,
            'Total return %': round((equity.iloc[-1] - 1) * 100, 2),
            'Total no. of trades': len(tradebook),
            'Winning trades': sum([1 for t in trades if t['pnl'] > 0]),
            'Max drawdown': strat.analyzers.drawdown.get_analysis()['max']['drawdown'],
            'Sharpe ratio': strat.analyzers.sharpe.get_analysis().get('sharperatio', 0),
            'Annual return': annual_return,
            'Annual volatility': annual_volatility,
            'Alpha': alpha,
            'Beta': beta,
            'Return of stock in backtested timeframe': (self.data._dataname['Close'].iloc[-1] / self.data._dataname['Close'].iloc[0]) - 1
            #'Return of stock in backtested timeframe': self.data.close[-1] / self.data.close[0] - 1
        }

        # Save chart
        if self.save_chart:
            fig = cerebro.plot(style='candlestick')[0][0]
            chart_path = f"output/charts/{self.symbol.replace(':', '_')}.png"
            os.makedirs(os.path.dirname(chart_path), exist_ok=True)
            fig.savefig(chart_path)

        return metrics, tradebook
