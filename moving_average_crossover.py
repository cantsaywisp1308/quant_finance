import numpy as np


def generate_signals(prices, short_window=20, long_window=50):
    """
    Generate trading signals based on moving average crossover.

    Returns:
        signals: array where 1 = buy, -1 = sell, 0 = hold
    """
    n = len(prices)
    signals = np.zeros(n)

    # Calculate moving averages
    short_ma = np.zeros(n)
    long_ma = np.zeros(n)

    for i in range(n):
        if i >= short_window - 1:
            short_ma[i] = np.mean(prices[i - short_window + 1:i + 1])
        if i >= long_window - 1:
            long_ma[i] = np.mean(prices[i - long_window + 1:i + 1])

    # TODO: Generate signals
    # When short_ma crosses ABOVE long_ma -> buy signal (1)
    # When short_ma crosses BELOW long_ma -> sell signal (-1)

    for i in range(long_window, n):
        # YOUR CODE HERE: detect crossovers
        if (short_ma[i] > long_ma[i] and short_ma[i - 1] <= long_ma[i - 1]):
            signals[i] = 1

        elif (short_ma[i] <= long_ma[i] and short_ma[i - 1] > long_ma[i]):
            signals[i] = -1
        pass

    return signals, short_ma, long_ma