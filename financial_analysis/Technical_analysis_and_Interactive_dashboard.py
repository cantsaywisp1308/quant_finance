import yfinance as yf
import pandas as pd
import talib
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.core.display_functions import display
from ta import add_all_ta_features
import mplfinance as mpf


def get_technical_data(stock, start_date, end_date):
    df = yf.download(stock,
                     start=start_date,
                     end=end_date,
                     progress=False,
                     auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):  # make a dataframe to be a ndarray
        df.columns = df.columns.get_level_values(0)

    # Calculate and plot the SMA_20 (Simple Moving Average)
    df["sma_20"] = talib.SMA(df["Close"], timeperiod=20)
    # (
    #     df[["Close", "sma_20"]]
    #     .plot(title="20-day Simple Moving Average (SMA)")
    # )

    # Calculate and plot the Bollinger bands
    df['bb_up'], df['bb_mid'], df['bb_low'] = talib.BBANDS(df["Close"])
    # fig, ax = plt.subplots()
    # (df.loc[:, ['Close','bb_up','bb_mid','bb_low']].plot(ax=ax, title="Bollinger Bands"))
    # ax.fill_between(df.index, df['bb_low'], df['bb_up'], color='gray', alpha=0.4)

    # Calculate and plot RSI
    df['rsi'] = talib.RSI(df['Close'])
    # fig, ax = plt.subplots()
    # df['rsi'].plot(ax=ax, title='Relative Strength Index (RSI)')
    # ax.hlines(y=30,
    #           xmin= df.index.min(),
    #           xmax=df.index.max(),
    #           colors='r')
    # ax.hlines(y=70,
    #           xmin=df.index.min(),
    #           xmax=df.index.max(),
    #           colors='r')

    # Calculate and plot the MACD
    df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    with sns.plotting_context("notebook"):
        fig, ax = plt.subplots(2, 1, sharex=True)

        (
            df[["macd", "macdsignal"]].
            plot(ax=ax[0],
                 title="Moving Average Convergence Divergence (MACD)")
        )
        ax[1].bar(df.index, df["macdhist"].values, label="macd_hist")
        ax[1].legend()
    sns.despine()
    plt.tight_layout()
    plt.show()

def technical_analysis_with_ta(stock, start_date, end_date):
    df = yf.download(stock,
                     start=start_date,
                     end=end_date,
                     progress=False,
                     auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):  # make a dataframe to be a ndarray
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = add_all_ta_features(df, open="Open", high="High",
                             low="Low", close="Close",
                             volume="Volume")
    print(df.columns)


def candlestick_patterns(stock, period, interval):
    df = yf.download(stock,period= period,interval= interval,progress=False,auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df['3_line_strike'] = talib.CDL3LINESTRIKE(df["Open"], df["High"], df["Low"], df["Close"])

    #locate and plot the bearish patterns
    # df[df["3_line_strike"] == -100].head().round(2)
    # print(df[df["3_line_strike"] == -100].head().round(2))
    # mpf.plot(df["2025-05-21 02:00:00":"2025-05-21 16:00:00"],
    #          type="candle")

    # locate and plot the bullish patterns
    # df[df['3_line_strike'] == 100].head().round(2)
    # print(df[df['3_line_strike'] == 100].head().round(2))
    # mpf.plot(df["2025-05-21 17:00:00":"2025-05-22 05:00:00"],
    #          type="candle")

    # Get all available pattern names
    candle_names = talib.get_function_groups()["Pattern Recognition"]
    for name in candle_names:
        df[name] = getattr(talib, name)(df["Open"], df["High"], df["Low"], df["Close"])
    with pd.option_context('display.max_rows', len(candle_names)):
        display(df[candle_names].describe().transpose().round(2))

    print("===============================================================")
    # Locate and plot the "Evening star" pattern
    print(df[df['CDLEVENINGSTAR'] == -100].head())
    mpf.plot(df["2025-05-23 15:00:00":"2025-05-24 05:00:00"], type="candle",)
    # print(df)

if __name__ == "__main__":
    # get_technical_data("AAPL", "2020-01-01", "2020-12-31")
    # technical_analysis_with_ta("AAPL", "2019-01-01", "2020-12-31")
    candlestick_patterns('BTC-USD', '9mo', '1h')