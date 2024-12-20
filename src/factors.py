import pandas as pd
import numpy as np


def VWAP(data, time_period=10): #data:pd.series
    factor = (data['last'] * data['volume']).rolling(time_period, min_periods=2).sum() / data['volume'].rolling(
        time_period, min_periods=2).sum()
    return factor

def MOM(data, time_period=10):
    """
    Calculate Momentum (MOM) of a stock's close prices over a specified time period.
    """
    # Calculate the difference in close prices between current time and 'timeperiod' periods ago
    mom_values = data['last'].apply(lambda x: (x - x.shift(time_period))/x.shift(time_period))
    return mom_values


def SMA(data, time_period=10):
    """
    Return Simple Moving Average
    """
    return data.rolling(time_period).mean()


def RSI(data, time_period=14):
    """
    Calculate the Relative Strength Index (RSI) of a stock's close prices over a specified time period.

    """
    # Calculate the price changes (daily returns)
    delta = data.diff()

    # Calculate the gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate the average gains and losses over the specified time period
    avg_gain = gain.rolling(window=time_period).mean()
    avg_loss = loss.rolling(window=time_period).mean()

    # Calculate the relative strength (RS) and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def MACD(factor, data, short_period=12, long_period=26, signal_period=9):
    """
    Calculate the Moving Average Convergence/Divergence (MACD) of a stock's close prices.
    """
    # Calculate the short-term and long-term EMAs
    short_ema = data.ewm(span=short_period, adjust=False).mean()
    long_ema = data.ewm(span=long_period, adjust=False).mean()

    # Calculate the MACD line (difference between short and long EMAs)
    macd_line = short_ema - long_ema

    # Calculate the signal line (EMA of the MACD line)
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    # Create a DataFrame to store MACD and Signal line values
    #macd_df = pd.DataFrame({'MACD': macd_line, 'Signal': signal_line})

    factor['macd'] = macd_line
    factor['signal'] = signal_line

    return factor #macd_df


def OBV(close, volume):
    """
    Calculate the On-Balance Volume (OBV) of a stock.
    """
    # Calculate the price changes (daily returns)
    price_changes = close.diff()
    volume = np.log(volume + 10e-7)
    # Initialize the OBV with the first value
    obv = pd.Series(0.0, index=close.index)

    for loc, i in enumerate(price_changes.index[1:]):
        pre_i = price_changes.index[loc-1]
        if price_changes[i] > 0:
            obv[i] = obv[pre_i] + volume[i]
        elif price_changes[i] < 0:
            obv[i] = obv[pre_i] - volume[i]
        else:
            obv[i] = obv[pre_i]
    factor = obv.values
    return factor

def PE_ratio(mkt_cap, shares, close):
    """
    Calculate the Price Earnings Ratio of a stock.
    """
    # Calculate the price changes (daily returns) # Stock Price Per Share/Earning Per Share (EPS)
    pe = mkt_cap/shares / close
    return pe

def Reg_factors(factor, close, timeperiod=14, nbdev=1):
    """
    Calculate regression factors indicators of a stock's close prices.
    """
    # Calculate Linear Regression
    linearreg = close.rolling(window=timeperiod).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)

    # Calculate Linear Regression Angle
    linearreg_angle = np.arctan(linearreg) * (180 / np.pi)

    # Calculate Linear Regression Intercept
    linearreg_intercept = close.rolling(window=timeperiod).apply(lambda x: np.polyfit(range(len(x)), x, 1)[1], raw=True)

    # Calculate Linear Regression Slope
    linearreg_slope = close.rolling(window=timeperiod).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)

    # Calculate STDDEV (Standard Deviation)
    stddev = close.rolling(window=timeperiod).std() * nbdev

    # Calculate TSF (Time Series Forecast)
    tsf = close.rolling(window=timeperiod).mean()

    # Calculate VAR (Variance)
    var = close.rolling(window=timeperiod).var() * nbdev

    # Create a DataFrame with the calculated indicators
    indicators = pd.DataFrame({
        'linear_reg': linearreg,
        'linear_reg_angle': linearreg_angle,
        'intercept': linearreg_intercept,
        'slope': linearreg_slope,
        'stddev': stddev,
        'tsf': tsf,
        'var': var
    })

    factor.concat(indicators, axis=1)

    return factor #indicators

