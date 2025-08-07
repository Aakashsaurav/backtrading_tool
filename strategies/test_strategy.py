import backtrader as bt
import customIndicator

class MyStrategy(bt.Strategy):
    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.supertrend = customIndicator.Supertrend(self.data, period=10, multiplier=3.0)
        self.rs = customIndicator.RelativeStrength(self.data, comparative_data=self.datas[1], period=55)
        self.order = None

    def next(self):
        if not self.position:
            if self.rsi < 30:
                self.order = self.buy()
        elif self.rsi > 70:
            self.order = self.sell()
