import backtrader as bt

class Supertrend(bt.Indicator):
    lines = ('supertrend', 'direction', 'final_upperband', 'final_lowerband')
    params = (('period', 10), ('multiplier', 3.0))

    plotinfo = dict(subplot=False)
    plotlines = dict(
        supertrend=dict(color='green'),
        final_upperband=dict(color='red', linestyle='dashed'),
        final_lowerband=dict(color='green', linestyle='dashed')
    )

    def __init__(self):
        self.addminperiod(self.p.period)
        atr = bt.ind.ATR(self.data, period=self.p.period)

        self.median = (self.data.high + self.data.low) / 2
        self.basic_upperband = self.median + self.p.multiplier * atr
        self.basic_lowerband = self.median - self.p.multiplier * atr

    def next(self):
        i = len(self)

        if i <= self.p.period:
            self.lines.supertrend[0] = float('nan')
            self.lines.direction[0] = float('nan')
            self.lines.final_upperband[0] = float('nan')
            self.lines.final_lowerband[0] = float('nan')
            return

        close = self.data.close[0]
        prev_close = self.data.close[-1]

        prev_direction = self.lines.direction[-1] if len(self.lines.direction) > 1 else -1
        prev_fub = self.lines.final_upperband[-1] if len(self.lines.final_upperband) > 1 else 0
        prev_flb = self.lines.final_lowerband[-1] if len(self.lines.final_lowerband) > 1 else 0

        bub = self.basic_upperband[0]
        blb = self.basic_lowerband[0]

        if prev_direction == 1:
            flb = max(blb, prev_flb)
            fub = prev_fub

            if close < flb:
                direction = -1
                fub = bub
                flb = blb
            else:
                direction = 1
        else:
            fub = min(bub, prev_fub)
            flb = prev_flb

            if close > fub:
                direction = 1
                flb = blb
                fub = bub
            else:
                direction = -1

        self.lines.final_upperband[0] = fub
        self.lines.final_lowerband[0] = flb
        self.lines.direction[0] = direction
        self.lines.supertrend[0] = flb if direction == 1 else fub
"""
import numpy as np

class RelativeStrength(bt.Indicator):
    '''
    Relative Strength (RS) = (Stock / Stock[-period]) / (Benchmark / Benchmark[-period]) - 1
    Computed in __init__ for speed. Only aligns overlapping dates.
    '''
    lines = ('rs',)
    params = (
        ('comparative_data', None),  # Must be a DataFeed, not just a line
        ('period', 55),
    )
    plotinfo = dict(subplot=False)
    plotlines = dict(
        rs=dict(color='yellow')
    )

    def __init__(self):
        if not self.p.comparative_data:
            raise ValueError("comparative_data must be a full data feed (not just .close)")

        # Extract numpy arrays of close prices and datetimes
        stock_dates = np.array([bt.num2date(self.data.datetime[i]).date() for i in range(len(self.data))])
        bench_dates = np.array([bt.num2date(self.p.comparative_data.datetime[i]).date() for i in range(len(self.p.comparative_data))])

        stock_close = np.array([self.data.close[i] for i in range(len(self.data))])
        bench_close = np.array([self.p.comparative_data.close[i] for i in range(len(self.p.comparative_data))])

        # Find common dates
        common_dates = np.intersect1d(stock_dates, bench_dates)

        # Preallocate RS array
        rs_values = [float('nan')] * len(self.data)

        # Create mapping for fast lookup
        bench_date_to_index = {bt.num2date(self.p.comparative_data.datetime[i]).date(): i for i in range(len(self.p.comparative_data))}
        stock_date_to_index = {bt.num2date(self.data.datetime[i]).date(): i for i in range(len(self.data))}

        for date in common_dates:
            i_stock = stock_date_to_index[date]
            i_bench = bench_date_to_index[date]

            if i_stock >= self.p.period and i_bench >= self.p.period:
                stock_now = stock_close[i_stock]
                stock_past = stock_close[i_stock - self.p.period]
                bench_now = bench_close[i_bench]
                bench_past = bench_close[i_bench - self.p.period]

                if stock_past != 0 and bench_past != 0:
                    rs = (stock_now / stock_past) / (bench_now / bench_past) - 1
                    rs_values[i_stock] = rs

        # Load into line buffer
        self.lines.rs.array[:] = rs_values
"""
import backtrader as bt


class RelativeStrength(bt.Indicator):
    """
    Relative Strength (RS) Indicator

    RS = (Stock / Stock[-period]) / (Benchmark / Benchmark[-period]) - 1

    It compares the price performance of the current stock with a benchmark.
    Works only on dates where both stock and benchmark have matching data.
    """

    lines = ('rs',)
    params = (
        ('comparative_data', None),  # Benchmark data feed
        ('period', 55),
    )
    plotinfo = dict(subplot=True)
    plotlines = dict(rs=dict(color='orange', _name='RelativeStrength'))

    def __init__(self):
        if self.p.comparative_data is None:
            raise ValueError("comparative_data (benchmark data feed) must be provided")

        self.addminperiod(self.p.period + 1)

        # Precompute date to index mapping
        self.stock_date_idx = {}
        self.bench_date_idx = {}

    def prenext(self):
        # Build date-index mappings for both stock and benchmark during warm-up
        for i in range(len(self.data)):
            dt = bt.num2date(self.data.datetime[i]).date()
            self.stock_date_idx[dt] = i

        for i in range(len(self.p.comparative_data)):
            dt = bt.num2date(self.p.comparative_data.datetime[i]).date()
            self.bench_date_idx[dt] = i

    def next(self):
        dt = bt.num2date(self.data.datetime[0]).date()

        if dt not in self.bench_date_idx:
            self.lines.rs[0] = float('nan')
            return

        i_stock = self.stock_date_idx.get(dt)
        i_bench = self.bench_date_idx.get(dt)

        if i_stock is None or i_bench is None:
            self.lines.rs[0] = float('nan')
            return

        if i_stock < self.p.period or i_bench < self.p.period:
            self.lines.rs[0] = float('nan')
            return

        # Get current and past prices
        stock_now = self.data.close[0]
        stock_past = self.data.close[-self.p.period]

        bench_now = self.p.comparative_data.close[0]
        bench_past = self.p.comparative_data.close[-self.p.period]

        if stock_past == 0 or bench_past == 0:
            self.lines.rs[0] = float('nan')
            return

        # RS formula
        rs = (stock_now / stock_past) / (bench_now / bench_past) - 1
        self.lines.rs[0] = rs
