
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm, probplot

class stockReturns:

    def __init__(self, stock, start_date, end_date):
        self.data = None
        self.stock = stock
        self.start_date = start_date
        self.end_date = end_date

    def download_data(self):
        data = {}
        # Ensure we get 'Adj Close' if possible
        ticker = yf.download(self.stock, start=self.start_date, end=self.end_date, auto_adjust=False, progress=False)
        
        val = None
        if 'Adj Close' in ticker:
            val = ticker['Adj Close']
        elif 'Close' in ticker:
            val = ticker['Close']
        else:
             raise ValueError(f"No price data found for {self.stock}")

        # If val is a DataFrame (MultiIndex or single col), convert to Series
        if isinstance(val, pd.DataFrame):
            val = val.iloc[:, 0]
            
        data[self.stock] = val
        return pd.DataFrame(data)

    def initialize(self):
        stock_data = self.download_data()
        # stock_data = stock_data.resample('ME').last()
        # Correctly calculate monthly return
        # stock_data[self.stock] is the price series
        stock_data['daily_return'] = np.log(stock_data[self.stock] / stock_data[self.stock].shift(1))

        self.data = stock_data[1:]
        print(self.data)
        #self.plot_monthly_returns()
        self.show_histogram(stock_data['daily_return'])
        # self.show_qq_plot(stock_data['daily_return'])

    # def show_qq_plot(self, stock_data):
    #     stock_data = stock_data.dropna()
    #     plt.figure(figsize=(10, 6))
    #     probplot(stock_data, plot=plt)
    #     plt.title('Q-Q Plot')
    #     plt.show()


    def show_histogram(self, stock_data):
        stock_data = stock_data.dropna()
        plt.hist(stock_data, bins=700, density=True)
        stock_variance = stock_data.var()
        stock_mean = stock_data.mean()
        sigma = np.sqrt(stock_variance)
        x = np.linspace(stock_mean - 3 * sigma, stock_mean + 3 * sigma, 100)
        plt.plot(x, norm.pdf(x, stock_mean, sigma))
        plt.ylabel('Density')
        plt.show()

if __name__ == "__main__":
    stockReturns = stockReturns('AMZN', '2017-12-01', '2025-01-01')
    stockReturns.initialize()