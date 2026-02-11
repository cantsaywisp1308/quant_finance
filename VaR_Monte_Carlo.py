import numpy as np
import pandas as pd
import yfinance as yf
import datetime

def download_data(stock, start_date, end_date):
    data = {}
    val = None
    ticker = yf.download(stock, start=start_date, end=end_date, auto_adjust=False, progress=False)
    if 'Adj Close' in ticker:
        val = ticker['Adj Close']
    else:
        raise ValueError(f"No price data found for {stock}")

    # If val is a DataFrame (MultiIndex or single col), convert to Series
    if isinstance(val, pd.DataFrame):
        val = val.iloc[:, 0]

    data['Adj Close'] = val
    return pd.DataFrame(data)

class ValueAtRiskMonteCarlo:

    def __init__(self, investment, mu, sigma, confidence, days, iterations):
        #value at the time t = 0 (for example: $1000)
        self.investment = investment
        self.mu = mu
        self.sigma = sigma
        self.confidence = confidence
        self.days = days
        self.iterations = iterations

    def simulation(self):
        rand = np.random.normal(0, 1, [1, self.iterations])
        
        #equation for the S(t) price
        #this is the random walk of the initial investment
        stock_price = self.investment * np.exp(self.days * (self.mu - 0.5 * self.sigma ** 2) + self.sigma * np.sqrt(self.days) * rand)

        #sort the stock price from smallest to the largest
        stock_price = np.sort(stock_price)

        #it depends on the confidence level (95% --> 5% or 99% --> 1%)
        percentile = np.percentile(stock_price, (1- self.confidence) * 100)

        return self.investment - percentile


if __name__ == '__main__':
    investment = 1e6
    confidence = 0.95
    days = 1
    iterations = 10000

    #historical data to approximate mean and standard deviation
    start_date = datetime.datetime(2014, 1, 1)
    end_date = datetime.datetime(2020, 12, 31)
    #download stock data from yahoo finance
    citi = download_data('C', start_date, end_date)
    print(citi)
    citi['returns'] = citi['Adj Close'].pct_change()

    #calculate the mean and standard deviation
    mu = np.mean(citi['returns'])
    sigma = np.std(citi['returns'])
    
    model = ValueAtRiskMonteCarlo(investment, mu, sigma,confidence,days,iterations)
    print('Value at risk with Monte Carlo simulation: $%.2f' % model.simulation())
