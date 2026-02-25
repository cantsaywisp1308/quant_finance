import yfinance as yf
import numpy as np
import pandas as pd
import pandas_datareader.data as web

import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web

import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
from datetime import datetime, timedelta


def download_data(stock):
    start_date = '2014-01-01'
    end_date = '2020-12-31'

    # 1. Download Stock Data
    df = yf.download(stock, start=start_date, end=end_date, progress=False, auto_adjust=False)

    # 2. Download CPI Data starting ONE MONTH EARLIER
    # This ensures we have a value to ffill into the start of 2014
    cpi_start = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=32)).strftime('%Y-%m-%d')
    cpi = web.DataReader('CPIAUCSL', 'fred', start=cpi_start, end=end_date)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 3. Calculate Returns
    df = df.resample("M").last()
    df['simple_return'] = df['Adj Close'].pct_change()
    df['log_return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))

    # 4. Use merge_asof for perfect alignment
    # merge_asof is smarter than join; it looks backward for the last available value
    df = df.sort_index()
    cpi = cpi.sort_index()

    # We use merge_asof to align the daily stock index with the monthly CPI index
    df = pd.merge_asof(df, cpi, left_index=True, right_index=True, direction='backward')

    # 5. Clean up
    df.rename(columns={'CPIAUCSL': 'cpi'}, inplace=True)
    df = df.dropna(subset=['simple_return', 'cpi'])
    df['inflation_rate'] = df['cpi'].pct_change()
    df['real_return'] = (df['simple_return'] + 1) / (df['inflation_rate'] + 1) -1
    print(df)
    return df


# Example usage:
# data = download_data('AAPL')
# print(data[['Adj Close', 'cpi']].head(15))


if __name__ == '__main__':
    download_data('AAPL')