import numpy as np
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import title

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader.data as web
from scipy.stats import alpha


def download_data(stock):
    start_date = '2020-01-01'
    end_date = '2020-12-31'

    # FIX 1: Initialize as a DataFrame, not a dictionary {}
    data = pd.DataFrame()

    ticker = yf.download(stock, start=start_date, end=end_date, auto_adjust=False, progress=False)

    if 'Adj Close' in ticker:
        # ticker['Adj Close'] might be a Series or DataFrame depending on yfinance version
        val = ticker['Adj Close']
    else:
        raise ValueError(f"No price data found for {stock}")

    # Ensure we are working with a Series
    if isinstance(val, pd.DataFrame):
        val = val.iloc[:, 0]

    # FIX 2: Assign columns to the DataFrame
    data['Adj Close'] = val
    data['simple_returns'] = data['Adj Close'].pct_change()

    #-----------------------------------------------------------------------

    sp500_data = pd.DataFrame()
    sp500_data = yf.download('^GSPC', start=start_date, end=end_date, auto_adjust=False, progress=False)

    # Now this multi-column selection will work perfectly!
    data[['Adj Close', 'simple_returns']].plot(subplots=True,
                                               sharex=True,
                                               title=f"{stock} stock in 2020")
    sns.despine()
    plt.tight_layout()
    plt.show()

    return data  # Return the dataframe so you can use it later


def download_unemployment_rate():
    unemployment = web.DataReader('UNRATE', 'fred', start='2014-01-01', end='2019-12-31')
    unemployment = unemployment.resample('ME').mean()
    # unemployment['UNRATE'].plot(title='Unemployment Rate from 2010 to 2020')
    # sns.despine()
    # plt.tight_layout()
    # plt.show()
    # print(unemployment)

    unemployment['month'] = unemployment.index.month
    unemployment['year'] = unemployment.index.year
    sns.lineplot(data=unemployment,
                 x="month",
                 y="UNRATE",
                 hue="year",
                 style="year",
                 legend="full",
                 palette="colorblind")

    plt.title("Unemployment rate - Seasonal plot")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2);

    sns.despine()
    plt.tight_layout()
    plt.show()

def compare_stock_and_SP500(stock, start_date='2025-01-01', end_date='2025-12-31'):
    data = yf.download([stock, 'SPY'], start=start_date, end=end_date, auto_adjust=False, progress=False)['Adj Close']

    #calculating daily returns
    daily_returns = data.pct_change()
    #calculate cumulative returns, shows the growth of $1 invested at the start date
    cum_returns = (1 + daily_returns).cumprod()

    plt.figure(figsize=[8,8])
    sns.lineplot(data=cum_returns)

    plt.title(f"Cumulative Returns - {stock} stock vs S&P500 in 2020")
    plt.ylabel("Growth of investment")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.legend([stock, 'S&P 500 (SPY)'])

    sns.despine()
    plt.show()

    #analyze final comparison
    #this is the formula of total percentage of returns
    final_perf = (cum_returns.iloc[-1] -1) * 100
    print(f"Total Return for {stock}: {final_perf[stock]:.2f}%")
    print(f"Total Return for S&P 500: {final_perf['SPY']:.2f}%")



if __name__ == '__main__':
    # download_data('MSFT')
    # download_unemployment_rate()
    compare_stock_and_SP500('AAPL')