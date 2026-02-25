import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import warnings

from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

def download_data():
    df = yf.download('AAPL', start='2014-01-01', end='2020-12-31', progress=False)
    print(df)

def download_data_ticker(stock):
    stock_data = yf.Ticker(stock)
    #print(stock_data.quarterly_financials)
    #print(stock_data.history())
    #print(stock_data.financials)
    # print(stock_data.income_stmt)
    print(stock_data.calendar)

if __name__ == '__main__':
    download_data_ticker('AAPL')