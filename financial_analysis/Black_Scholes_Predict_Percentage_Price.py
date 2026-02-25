import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


def predict_asset_monte_carlo(ticker, days_to_predict=252, num_sims=1000):
    # 1. Pull Historical Data - Added 'auto_adjust' to simplify columns
    df_raw = yf.download(ticker, period='1y', auto_adjust=True)
    if df_raw.empty: return print("Ticker not found.")

    # Ensure we are looking at a 1D Series of prices
    data = df_raw['Close'].squeeze()

    # 2. Calculate Parameters - Force to float values
    log_returns = np.log(data / data.shift(1)).dropna()
    mu = float(log_returns.mean())
    sigma = float(log_returns.std())
    S0 = float(data.iloc[-1])

    # 3. Simulation Logic (Vectorized)
    # This creates a matrix of [days x simulations]
    random_shocks = np.random.normal(0, 1, (days_to_predict, num_sims))
    drift = (mu - 0.5 * sigma ** 2)
    daily_returns = np.exp(drift + sigma * random_shocks)

    # Cumulative product along the days axis to get price paths
    # We start with S0 and multiply through the returns
    price_paths = S0 * daily_returns.cumprod(axis=0)

    # 4. Analysis
    mean_path = np.mean(price_paths, axis=1)
    lower_bound = np.percentile(price_paths, 5, axis=1)
    upper_bound = np.percentile(price_paths, 95, axis=1)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(price_paths[:, :50], color='gray', alpha=0.2)  # Show sample paths
    plt.plot(mean_path, color='blue', linewidth=2, label='Expected Mean')
    plt.fill_between(range(days_to_predict), lower_bound, upper_bound, color='blue', alpha=0.1,
                     label='90% Confidence Interval')
    plt.title(f"Monte Carlo: {ticker} Future Price Probability")
    plt.legend()
    plt.show()

    print(f"Current Price: ${S0:.2f}")
    print(f"Predicted Mean after {days_to_predict} days: ${mean_path[-1]:.2f}")


if __name__ == '__main__':
    predict_asset_monte_carlo('NVDA')