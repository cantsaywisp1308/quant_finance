import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.ndimage import label

RISK_FREE_RATE = 0.05
MONTH_OF_THE_YEAR = 12
class CAPM :

    def __init__(self, stocks, start_date, end_date):
        self.data = None
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date

    def download_data(self):
        data = {}
        for stock in self.stocks:
            ticker = yf.download(stock, self.start_date, self.end_date, auto_adjust=False, progress=False)
            s = ticker['Adj Close']
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
                data[stock] = s
            # if 'Adj Close' in ticker:
            #     s = ticker['Adj Close']
            #     if isinstance(s, pd.DataFrame):
            #         s = s.iloc[:, 0]
            #     data[stock] = s
            # elif 'Close' in ticker:
            #     s = ticker['Close']
            #     if isinstance(s, pd.DataFrame):
            #         s = s.iloc[:, 0]
            #     data[stock] = s
            # else:
            #     raise ValueError(f"No price data found for {stock}")

        #Now yfinance return data of a stock as DataFrame with multiple columns so we have to pass s = s.iloc[:, 0] as a one-index column DataFrame
        return pd.DataFrame(data)

    def initialize(self):
        stock_data = self.download_data()
        #use monthly return not daily return  like in the Markovitz model
        stock_data = stock_data.resample('ME').last()

        self.data = pd.DataFrame({'s_adjClose' : stock_data[self.stocks[0]],
                                  'm_adjClose' : stock_data[self.stocks[1]],})
        self.data[['s_return', 'm_return']] = np.log(self.data[['s_adjClose', 'm_adjClose']] /
                                                     self.data[['s_adjClose', 'm_adjClose']].shift(1))
        self.data = self.data[1:]
        print(self.data)

    def calculate_beta(self):
        #calculate covariance matrix: the diagonal items are the variance
        #off diagonals are the covariance
        #the matrix is symmetric: cov[0:1] = cov[1:0]
        covariance_matrix = np.cov(self.data['s_return'], self.data['m_return'])
        #calculate beta according to the formula
        beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]
        #beta is the risk of the portfolio to the market
        print("Beta from formula: ", beta)
        print(covariance_matrix)
        #Beta = 1: stock go with the market; beta > 1:market is riskier than the stock; beta < 1: market risk is lower than the stock

    def regression(self):
        #using linear regression to fit the line of the data
        #[stock_returns, market_returns] - slope is the beta
        beta, alpha = np.polyfit(self.data['m_return'], self.data['s_return'], 1)
        print("Beta from regression: ", beta)
        #calculate the expect return according to CAPM formula
        #we are after annual return (we multiply by 12)
        expect_return = RISK_FREE_RATE + beta * (self.data['m_return'].mean() * MONTH_OF_THE_YEAR -
                                                 RISK_FREE_RATE)
        print("Expected Return: ", expect_return)
        self.plot_regression(alpha, beta)

    def plot_regression(self, alpha, beta):
        fig, axis = plt.subplots(1, figsize=(20, 10))
        axis.scatter(self.data['m_return'], self.data['s_return'], label='Data Points')
        axis.plot(self.data['m_return'], beta * self.data['m_return'] + alpha, color='red', label = 'CAPM Line')
        plt.title('Capital Asset Pricing Model , finding alpha and beta')
        plt.xlabel('Market Return $R_m$', fontsize=18)
        plt.ylabel('Stock Return $R_a$', fontsize=18)
        plt.text(0.08, 0.5, r'$R_a = \beta * R_m + \alpha$', fontsize=18)
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    capm = CAPM(['IBM', '^GSPC'], '2017-12-01', '2025-01-01')
    capm.initialize()
    capm.calculate_beta()
    capm.regression()