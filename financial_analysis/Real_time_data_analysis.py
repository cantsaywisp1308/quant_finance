import yfinance as yf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sktime.transformations.series.outlier_detection import HampelFilter
import ruptures as rpt
import pymannkendall as mk
import scipy.stats as scs
import statsmodels.api as sm
import statsmodels.tsa.api as smt


def download_data(stock):
    df = yf.download(stock, start='2019-01-01', end='2020-12-31', progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df["rtn"] = df["Adj Close"].pct_change()
    #df = df[["rtn"]].copy()
    print(df[20:])

    # #define mean and standard deviation windows = 21
    # df_rolling = df[["rtn"]].rolling(window=21) \
    #     .agg(["mean", "std"])
    # df_rolling.columns = df_rolling.columns.droplevel()
    # df = df.join(df_rolling)
    #
    # #calculate the upper and lower point
    # N_SIGMA = 3
    # df['upper'] = df['mean'] + N_SIGMA * df['std']
    # df['lower'] = df['mean'] - N_SIGMA * df['std']
    #
    # #define outliers
    # df['outlier'] = (df['rtn'] > df['upper']) | (df['rtn'] < df['lower'])
    #
    # #plot the data
    # fig, ax = plt.subplots()
    # df[["rtn","lower","upper"]].plot(ax=ax)
    # ax.scatter(df.loc[df['outlier']].index,
    #            df.loc[df['outlier'], 'rtn'],
    #            color='black', label='outlier')
    # ax.set_title(f"{stock}'s returns")
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # sns.despine()
    # plt.tight_layout()
    # plt.show()

    return df

def detect_outliers(stock, df, column, windows_size, n_sigma):
    df = df[[column]].copy()
    df_rolling = df[[column]].rolling(window=windows_size).agg(['mean', 'std'])
    df_rolling.columns = df_rolling.columns.droplevel()
    df = df.join(df_rolling)
    df['upper'] = df['mean'] + n_sigma * df['std']
    df['lower'] = df['mean'] - n_sigma * df['std']

    df['outlier'] = (df['rtn'] > df['upper']) | (df['rtn'] < df['lower'])
    #plot the data
    fig, ax = plt.subplots()
    df[["rtn","lower","upper"]].plot(ax=ax)
    ax.scatter(df.loc[df['outlier']].index,
               df.loc[df['outlier'], 'rtn'],
               color='black', label='outlier')
    ax.set_title(f"{stock}'s returns")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    sns.despine()
    plt.tight_layout()
    plt.show()
    return ((df[column] > df['upper']) | df[column] < df['lower'])

def detect_outlier_with_Hampel_filter(stock, start_date, end_date):
    df = yf.download(stock, start=start_date, end=end_date, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df["rtn"] = df["Adj Close"].pct_change()
    hampel_detector = HampelFilter(window_length=10, return_bool=True)
    df['outlier'] = hampel_detector.fit_transform(df['Adj Close'])
    fig, ax = plt.subplots()

    df[["Adj Close"]].plot(ax=ax)
    ax.scatter(df.loc[df["outlier"]].index,
               df.loc[df["outlier"], "Adj Close"],
               color="black", label="outlier")
    ax.set_title(f"{stock}'s stock price")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    sns.despine()
    plt.tight_layout()
    plt.show()
    print(df)

def detect_significant_change_points(symbol, start_date, end_date, penalty=70, smooth_days=5):
    """
    detects only significant structural shifts by smoothing data and increasing penalty.
    """
    # 1. Download & Clean
    df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Adj Close']].reset_index()
    df.columns = ['time', 'price']

    # 2. SMOOTHING: Apply a moving average to ignore daily volatility
    # This is the "secret sauce" for cleaner stock change points
    df['smoothed_price'] = df['price'].rolling(window=smooth_days).mean()

    # Drop the NaN values created by the rolling window
    clean_df = df.dropna().reset_index(drop=True)
    signal = clean_df['smoothed_price'].values

    # 3. DETECTION: Using a higher penalty for "Significance"
    # We use 'l2' (squared error) to find shifts in the mean price
    algo = rpt.Pelt(model="l2").fit(signal)
    result = algo.predict(pen=penalty)

    # 4. Visualization
    plt.figure(figsize=(14, 7))
    plt.plot(df['time'], df['price'], label='Actual Price', alpha=0.3, color='gray')
    plt.plot(clean_df['time'], clean_df['smoothed_price'], label=f'{smooth_days}-Day SMA', color='royalblue',
             linewidth=2)

    for cp_index in result[:-1]:
        cp_time = clean_df['time'].iloc[cp_index]
        plt.axvline(x=cp_time, color='red', linestyle='--', linewidth=2)
        print(f"Significant change point: {cp_time.date()}")

    plt.title(f"Significant Trend Shifts: {symbol} (Penalty={penalty})")
    plt.legend()
    plt.show()

    return df, result

def detect_stock_trends(stock, start_date, end_date, window_size=30):
    """
        Identifies windows of time where a statistically significant
        upward or downward trend exists.
    """
    df = yf.download(stock, start=start_date, end=end_date, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Adj Close']].reset_index()
    df.columns = ['time', 'price']

    result = []
    # 2. Rolling Window Trend Detection
    # We slide a window across the data, just like window_size=30 in Kats
    for i in range(len(df) - window_size):
        windows_data = df['price'].iloc[i : i + window_size]
        #perform Mann-Kendall Test
        test_result = mk.original_test(windows_data)

        # test_result.trend: 'increasing', 'decreasing', or 'no trend'
        # test_result.p: p-value (significance)
        if test_result.trend != 'no trend':
            result.append({
                'start_time': df['time'].iloc[i],
                'end_time': df['time'].iloc[i + window_size],
                'trend': test_result.trend,
                'p_value': test_result.p,
                'slope': test_result.slope
            })

    #3. Visualization
    plt.figure(figsize=(14, 7))
    plt.plot(df['time'], df['price'], label='Price', color='black', alpha=0.6)
    # Highlight upward trends in Green and downward in Red
    for res in result:
        color = 'green' if res['trend'] == 'increasing' else 'red'
        plt.axvspan(res['start_time'], res['end_time'], color=color, alpha=0.1)
    plt.title(f"Mann-Kendall Trend Analysis: {stock} ({window_size}-day window)")
    plt.ylabel("Price")
    plt.legend(['Price', 'Upward Trend', 'Downward Trend'])
    plt.show()

    return pd.DataFrame(result)

def detect_pattern_Hurst_exponent():
    df = yf.download("^GSPC",
                     start="2000-01-01",
                     end="2019-12-31",
                     progress=False,
                     auto_adjust=False)
    df["Adj Close"].plot(title="S&P 500 (years 2000-2019)")
    for lag in [20, 100, 250, 500, 1000]:
        hurst_exp = get_Hurst_exponent(df["Adj Close"].values, lag)
        print(f"Hurst exponent with {lag} lags: {hurst_exp:.4f}")
    # sns.despine()
    # plt.tight_layout()
    # plt.show()

def get_Hurst_exponent(ts, max_lag=20):
    """Returns the Hurst Exponent of the time series"""
    lags = range(2, max_lag)
    # standard deviations of the lagged differences
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    # calculate the slope of the log plot -> the Hurst Exponent
    hurst_exponent = np.polyfit(np.log(lags), np.log(tau), 1)[0]
    return hurst_exponent

def stylized_fact_asset_return():
    df = yf.download("^GSPC",
                     start="2000-01-01",
                     end="2020-12-31",
                     progress=False,
                     auto_adjust=False)
    df = df[["Adj Close"]].rename(
        columns={"Adj Close": "adj_close"}
    )
    df["log_rtn"] = np.log(df["adj_close"] / df["adj_close"].shift(1))
    df = df[["adj_close", "log_rtn"]].dropna()
    r_range = np.linspace(min(df["log_rtn"]), max(df["log_rtn"]), num=1000)
    mu = np.mean(df["log_rtn"])
    sigma = np.std(df["log_rtn"])
    norm_pdf = scs.norm.pdf(r_range, loc=mu, scale=sigma)
    # fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    # histogram
    # sns.distplot(df.log_rtn, kde=False,
    #              norm_hist=True, ax=ax[0])
    # ax[0].set_title("Distribution of S&P 500 returns",
    #                 fontsize=16)
    # ax[0].plot(r_range, norm_pdf, "g", lw=2,
    #            label=f"N({mu:.2f}, {sigma ** 2:.4f})")
    # ax[0].legend(loc="upper left");

    # Q-Q plot
    # qq = sm.qqplot(df.log_rtn.values, line="s", ax=ax[1])
    # ax[1].set_title("Q-Q plot", fontsize=16)
    #
    # sns.despine()
    # plt.tight_layout()
    # plt.show()
    # jb_test = scs.jarque_bera(df["log_rtn"].values)
    #
    # print("---------- Descriptive Statistics ----------")
    # print("Range of dates:", min(df.index.date), "-", max(df.index.date))
    # print("Number of observations:", df.shape[0])
    # print(f"Mean: {df.log_rtn.mean():.4f}")
    # print(f"Median: {df.log_rtn.median():.4f}")
    # print(f"Min: {df.log_rtn.min():.4f}")
    # print(f"Max: {df.log_rtn.max():.4f}")
    # print(f"Standard Deviation: {df.log_rtn.std():.4f}")
    # print(f"Skewness: {df.log_rtn.skew():.4f}")
    # print(f"Kurtosis: {df.log_rtn.kurtosis():.4f}")
    # print(f"Jarque-Bera statistic: {jb_test[0]:.2f} with p-value: {jb_test[1]:.2f}")

    # df['log_rtn'].plot(title="Daily S&P 500 returns", figsize=(10, 6))
    # sns.despine()
    # plt.tight_layout()
    # plt.show()

    # N_LAGS = 50
    # SIGNIFICANCE_LEVEL = 0.05
    #
    # acf = smt.graphics.plot_acf(df['log_rtn'], lags=N_LAGS, alpha=SIGNIFICANCE_LEVEL)
    # sns.despine()
    # plt.tight_layout()
    # plt.show()

    #5. check leverage effect
    #1. Check volatility measure as moving standard deviations
    df['rolling_std_252'] = df[["log_rtn"]].rolling(window=252).std()
    df['rolling_std_21'] = df[["log_rtn"]].rolling(window=21).std()

    #2. plot the series
    fig, ax = plt.subplots(3,1,figsize=(18, 15), sharex=True)
    df['adj_close'].plot(ax=ax[0])
    ax[0].set(title="S&P 500 time series",
          ylabel="Price ($)")
    df['log_rtn'].plot(ax=ax[1])
    ax[1].set(title="Log returns")

    df['rolling_std_252'].plot(ax=ax[2], color='r', label="Rolling Standard Deviation 252")
    df['rolling_std_21'].plot(ax=ax[2], color='g', label="Rolling Standard Deviation 21")
    ax[2].set(ylabel='Moving volatility', xlabel='Date')
    ax[2].legend()

    sns.despine()
    plt.tight_layout()
    plt.show()


def vix_vs_sp500():
    df = yf.download(["^GSPC", "^VIX"],
                     start="2000-01-01",
                     end="2020-12-31",
                     progress=False,
                     auto_adjust=False)
    df = df[["Adj Close"]]
    df.columns = df.columns.droplevel(0)
    df = df.rename(columns={"^GSPC": "sp500", "^VIX": "vix"})
    df['log_rtn'] = np.log(df['sp500']/df['sp500'].shift(1))
    df['vol_rtn'] = np.log(df['vix']/df['vix'].shift(1))
    df.dropna(how='any', axis=0, inplace=True)
    corr_coeff = df['log_rtn'].corr(df['vol_rtn'])
    ax = sns.regplot(x = 'log_rtn', y = 'vol_rtn', data=df, line_kws={'color': 'red'})
    ax.set(title=f"S&P 500 vs. VIX ($\\rho$ = {corr_coeff:.2f})",
       ylabel="VIX log returns",
       xlabel="S&P 500 log returns")
    sns.despine()
    plt.tight_layout()
    plt.show()
    print(df)


if __name__ == '__main__':
    # df = download_data('TSLA')
    # print(detect_outliers('TSLA',df, 'rtn', 21, 3))
    # detect_outlier_with_Hampel_filter('TSLA', '2019-01-01', '2020-12-31')
    # data, cp_list = detect_significant_change_points('TSLA', '2020-01-01', '2020-12-31')
    # print(data)
    # trend_df = detect_stock_trends('AAPL', "2020-01-01", "2026-01-31", window_size=30)
    # print(trend_df)
    # detect_pattern_Hurst_exponent()
    # stylized_fact_asset_return()
    vix_vs_sp500()