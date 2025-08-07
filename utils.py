# utils.py
import yfinance as yf
import pandas as pd
import backtrader as bt
import os
import numpy as np

def calculate_cagr(equity_curve):
    duration_years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    return (equity_curve[-1] / equity_curve[0])**(1 / duration_years) - 1

def calculate_drawdown(equity_curve):
    cum_max = equity_curve.cummax()
    dd = (equity_curve - cum_max) / cum_max
    return dd.min()

def calculate_sharpe(returns, rf=0.0):
    excess = returns - rf
    return np.mean(excess) / np.std(excess) * np.sqrt(252) if np.std(excess) > 0 else 0

def load_data(symbol, start='2018-01-01', end=None, period=None, interval=None, is_bechmark=False):
    if period and interval:
        df = yf.download(symbol, period=period, interval=interval, multi_level_index=False, rounding=True,
             group_by='ticker',auto_adjust=False)
    elif start:
        df = yf.download(symbol, start=start, end=end, multi_level_index=False, rounding=True,
                         group_by='ticker',auto_adjust=False)
    
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    #df.columns = [c.capitalize() for c in df.columns]  # Adjust to Backtrader format
    
    if is_bechmark:
        df = df['Adj Close'].dropna()
        return df
    else:
        return bt.feeds.PandasData(dataname=df)
    
    return bt.feeds.PandasData(dataname=df)
    
def supertrend(df, period=10, multiplier=3):
    """
    Calculates the Supertrend indicator using vectorized operations for performance
    and corrects the Supertrend value calculation during trend changes.

    Args:
        df (pd.DataFrame): DataFrame containing OHLC price data with columns 'High', 'Low', 'Close'.
        period (int): The lookback period for calculating Average True Range (ATR).
        multiplier (float): The multiplier for the ATR to calculate the band distance.

    Returns:
        pd.DataFrame: A DataFrame with the Supertrend indicator ('Supertrend') and
                      the Supertrend direction ('Supertrend_Direction': 1 for uptrend, -1 for downtrend).
                      Returns an empty DataFrame if required columns are missing or not enough data.
    """
    if not all(col in df.columns for col in ['High', 'Low', 'Close']):
        print("Error: DataFrame must contain 'High', 'Low', and 'Close' columns.")
        return pd.DataFrame()

    df = df.copy()

    # Calculate ATR using talib
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=period)

    # Calculate Basic Upper and Lower Bands using vectorized operations
    df['median_price'] = (df['High'] + df['Low']) / 2
    df['Basic_Upper_Band'] = df['median_price'] + multiplier * df['ATR']
    df['Basic_Lower_Band'] = df['median_price'] - multiplier * df['ATR']

    # Initialize Final Bands and Supertrend Direction arrays
    final_upper_band = np.full(len(df), np.nan)
    final_lower_band = np.full(len(df), np.nan)
    supertrend_direction = np.full(len(df), np.nan)

    close_prices = df['Close'].values
    basic_upper_band_values = df['Basic_Upper_Band'].values
    basic_lower_band_values = df['Basic_Lower_Band'].values

    # Determine the first valid index after the ATR period
    first_valid_idx = period - 1

    # Initialize the first valid Supertrend direction and bands
    if close_prices[first_valid_idx] > basic_upper_band_values[first_valid_idx]:
        supertrend_direction[first_valid_idx] = 1 # Uptrend
        final_lower_band[first_valid_idx] = basic_lower_band_values[first_valid_idx]
        final_upper_band[first_valid_idx] = basic_upper_band_values[first_valid_idx] # Initialize upper band as well
    elif close_prices[first_valid_idx] < basic_lower_band_values[first_valid_idx]:
        supertrend_direction[first_valid_idx] = -1 # Downtrend
        final_upper_band[first_valid_idx] = basic_upper_band_values[first_valid_idx]
        final_lower_band[first_valid_idx] = basic_lower_band_values[first_valid_idx] # Initialize lower band as well
    else: # Price is between bands, determine initial direction based on first movement
         if close_prices[first_valid_idx] > close_prices[first_valid_idx - 1]:
             supertrend_direction[first_valid_idx] = 1
             final_lower_band[first_valid_idx] = basic_lower_band_values[first_valid_idx]
             final_upper_band[first_valid_idx] = basic_upper_band_values[first_valid_idx]
         else:
             supertrend_direction[first_valid_idx] = -1
             final_upper_band[first_valid_idx] = basic_upper_band_values[first_valid_idx]
             final_lower_band[first_valid_idx] = basic_lower_band_values[first_valid_idx]

    # Loop to calculate Final Bands and Supertrend Direction for subsequent periods
    for i in range(first_valid_idx + 1, len(df)):
        prev_direction = supertrend_direction[i-1]
        prev_final_upper_band = final_upper_band[i-1]
        prev_final_lower_band = final_lower_band[i-1]

        current_basic_upper_band = basic_upper_band_values[i]
        current_basic_lower_band = basic_lower_band_values[i]
        current_close_price = close_prices[i]

        if prev_direction == 1: # Previous trend was uptrend
            # Calculate current period's potential lower band
            current_final_lower_band = max(current_basic_lower_band, prev_final_lower_band)
            current_final_upper_band = prev_final_upper_band # Upper band stays the same in uptrend

            if current_close_price < current_final_lower_band: # Downtrend signal
                supertrend_direction[i] = -1
                # When flipping to downtrend, the Supertrend value is the current Basic Upper Band
                current_final_upper_band = current_basic_upper_band
                current_final_lower_band = current_basic_lower_band # Initialize lower band as well
            else: # Continue uptrend
                supertrend_direction[i] = 1

        elif prev_direction == -1: # Previous trend was downtrend
            # Calculate current period's potential upper band
            current_final_upper_band = min(current_basic_upper_band, prev_final_upper_band)
            current_final_lower_band = prev_final_lower_band # Lower band stays the same in downtrend


            if current_close_price > current_final_upper_band: # Uptrend signal
                supertrend_direction[i] = 1
                 # When flipping to uptrend, the Supertrend value is the current Basic Lower Band
                current_final_lower_band = current_basic_lower_band
                current_final_upper_band = current_basic_upper_band # Initialize upper band as well
            else: # Continue downtrend
                supertrend_direction[i] = -1

        # Assign the calculated final bands for the current period
        final_upper_band[i] = current_final_upper_band
        final_lower_band[i] = current_final_lower_band


    # Assign the calculated arrays back to the DataFrame
    df['Supertrend_Direction'] = supertrend_direction
    df['Final_Upper_Band'] = final_upper_band
    df['Final_Lower_Band'] = final_lower_band


    # Calculate the final Supertrend value based on direction
    # In an uptrend (direction 1), the Supertrend is the Final Lower Band.
    # In a downtrend (direction -1), the Supertrend is the Final Upper Band.
    df['Supertrend'] = np.where(df['Supertrend_Direction'] == 1, df['Final_Lower_Band'], df['Final_Upper_Band'])

    # Return the relevant columns, starting from the first valid data point
    return df[['Supertrend', 'Supertrend_Direction']].iloc[period-1:]

def relative_strength(base_df, comparative_df, period=55):
    rs = (base_df / base_df.shift(period)) / (comparative_df/comparative_df.shift(period)) - 1
    return rs

from scipy.signal import argrelextrema

def find_support_resistance(df, order=5):
    """
    Adds resistance (peak) and support (bottom) columns to the dataframe using argrelextrema.
    """
    df = df.copy()
    close = df['Close'].values

    peaks = argrelextrema(close, np.greater, order=order)[0]
    bottoms = argrelextrema(close, np.less, order=order)[0]

    df['Resistance'] = False
    df['Support'] = False

    df.iloc[peaks, df.columns.get_loc('Resistance')] = True
    df.iloc[bottoms, df.columns.get_loc('Support')] = True

    return df
