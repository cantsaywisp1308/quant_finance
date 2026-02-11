import numpy as np
import yfinance as yf
from scipy.stats import norm
import pandas as pd
import datetime

def download_data(stock, start_date, end_date):

    data = {}
    val = None
    ticker = yf.download(stock, start= start_date, end=end_date, auto_adjust=False, progress=False)
    if 'Adj Close' in ticker:
        val = ticker['Adj Close']
    else:
        raise ValueError(f"No price data found for {stock}")

    # If val is a DataFrame (MultiIndex or single col), convert to Series
    if isinstance(val, pd.DataFrame):
        val = val.iloc[:, 0]

    data[stock] = val
    return pd.DataFrame(data)

#assume that we calculate the VaR for tomorrow (or the maximum loss tomorrow)
def calculate_VaR(position, c, mu, sigma): #c: confidence level (95%) mu: average, sigma: standard deviation
    var = position * (mu - sigma * norm.ppf(1-c))
    return var

def calculate_VaR_n(position, c, mu, sigma, n): #c: confidence level (95%) mu: average, sigma: standard deviation
    var = position * (mu * n - sigma * np.sqrt(n) * norm.ppf(1-c))
    return var

if __name__ == '__main__':
    start = datetime.datetime(2014, 1, 1)
    end = datetime.datetime(2025, 12, 31)
    stock_data = download_data('C', start, end)

    stock_data['returns'] = np.log(stock_data['C'] / stock_data['C'].shift(1))
    stock_data = stock_data[1:]
    print(stock_data)

    #this is the investment (in stock, etc)
    S = 1e6
    #confidence level
    c = 0.95
    #assume that the returns is in normal distribution
    mu = np.mean(stock_data['returns'])
    sigma = np.std(stock_data['returns'])
    print('Value at risk is: $%.2f' % calculate_VaR(S, c, mu, sigma))

    n = 252
    print('Value at risk in 252 days is: $%.2f' % calculate_VaR_n(S, c, mu, sigma, n))


