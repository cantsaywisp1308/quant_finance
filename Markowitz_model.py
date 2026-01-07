import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimization
from IPython.core.pylabtools import figsize
from scipy.ndimage import label

#there are average 252 days of trading in a year
NUM_TRADING_DAYS = 252
#we will generate randomly w (different portfolio)
NUM_PORTFOLIO = 10000

#stocks that we are about to handle
stocks = ['AAPL', 'WMT', 'TSLA', 'GE', '`MZN', 'DB']

#analyze the historical data - START to END
start_date = '2012-01-01'
end_date = '2017-01-01'

def download_data():
    #name of the stock {key} - stock values (2010-2017) as values
    stock_data = {}
    for stock in stocks:
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start=start_date, end=end_date)['Close']

    #stock_data = pd.read_csv('stock_data.csv')

    dataset = pd.DataFrame(stock_data)
    dataset.to_csv('stock_data.csv')
    return pd.DataFrame(dataset)

def show_data(data):
    data.plot(figsize= (16, 8))
    plt.show()

def calculate_returns(data):
    log_return = np.log(data/data.shift(1)) # calculated by S(t+1)/S(t)
    #data1 = pd.DataFrame(log_return[1:])
    #data1.to_csv('log_return.csv')
    #print(log_return[1:])
    return log_return[1:]

def statistics(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * NUM_TRADING_DAYS, weights)))
    return np.array([portfolio_return, portfolio_volatility, portfolio_return/portfolio_volatility])

def show_statistics(returns):
    #instead of daily metrics we are after annual metrics
    #mean of annual returns
    print(returns.mean() * NUM_TRADING_DAYS)
    print(returns.cov() * NUM_TRADING_DAYS)

def show_mean_variance(returns, weight):
    #we are here after the annual return
    portfolio_return = np.sum(returns.mean() * weight) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot(weight.T ,np.dot(returns.cov() * NUM_TRADING_DAYS, weight))) #multiple 2 matrices, measure of how much an investment portfolio's value swings up and down over time
    print('Portfolio Return Mean (return):', portfolio_return)
    print('Portfolio Volatility (standard deviation):', portfolio_volatility)

def show_portfolios(returns, volatilities):
    plt.figure(figsize = (10, 6))
    plt.scatter(volatilities, returns, c = returns/volatilities, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.show()

def show_optimal_portfolio(opt, rets, portfolio_rets, portfolio_vols):
    plt.figure(figsize=(10, 6))
    plt.scatter(portfolio_vols, portfolio_rets, c=portfolio_rets / portfolio_vols, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.plot(statistics(opt['x'], rets)[1], statistics(opt['x'], rets)[0], 'g*', markersize=20.0)
    plt.show()


def generate_portfolio(returns):
    portfolio_means = []
    portfolio_risks = []
    portfolio_weights = []
    for _ in range(NUM_PORTFOLIO):
        w = np.random.random(len(stocks))
        w /= np.sum(w)
        portfolio_weights.append(w)
        portfolio_means.append(np.sum(returns.mean() * w) * NUM_TRADING_DAYS)
        portfolio_risks.append(np.sqrt(np.dot(w.T, np.dot(returns.cov()
                                                          * NUM_TRADING_DAYS, w))))

    return np.array(portfolio_weights), np.array(portfolio_means), np.array(portfolio_risks)



#the minimum of a function f(x) is -f(x)
def min_function_sharpe_ratio(weights, returns):
    return -statistics(weights, returns)[2]

#what are constraints, the sum of weight = 1
#sum w - 1 =0 f(x) = 0 means this is the function to minimize
def optimize_portfolios(weights, returns):
    constraints = {'type': 'eq', 'fun' : lambda x: np.sum(x) - 1}
    #when weights is 1, means that the 100% of the money is in a single stock
    bounds = tuple((0, 1) for _ in range(len(stocks)))
    return optimization.minimize(fun=min_function_sharpe_ratio, x0=weights[0], args=returns
                                 , method='SLSQP', bounds=bounds, constraints=constraints)


def print_optimal_portfolio(optimum, returns):
    print('Optimal Portfolio:', optimum['x'].round(3))
    print('Expected Return, volatility and Sharpe Ratio:', statistics(optimum['x'].round(3), returns))

if __name__ == '__main__':
    dataset = download_data()
    show_data(dataset)
    log_daily_returns = calculate_returns(dataset)
    #show_statistics(log_daily_returns)

    weights, means, risks = generate_portfolio(log_daily_returns)
    show_portfolios(means, risks)
    optimum = optimize_portfolios(weights, log_daily_returns)
    print_optimal_portfolio(optimum, log_daily_returns)
    show_optimal_portfolio(optimum, log_daily_returns, means, risks)