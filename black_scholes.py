from scipy import stats
from numpy import log, exp, sqrt

def call_option_price(S, K, T, r, sigma):

    #calculate d1 and d2
    d1 = (log(S/K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    return S * stats.norm.cdf(d1) - K * exp(-r * T) * stats.norm.cdf(d2)

def put_option_price(S, K, T, r, sigma):

    #calculate d1 and d2
    d1 = (log(S/K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    return -S * stats.norm.cdf(-d1) + K * exp(-r * T) * stats.norm.cdf(-d2)

if __name__ == '__main__':
    #underlying stock at T=0
    S0 = 100
    #strike price
    K = 100
    #expity date T = 1 (1 year = 365 days)
    T = 1
    #risk-free rate
    rf = 0.05
    #volatility of the underlying stock
    sigma = 0.2

    print("call option price according to Black-Scholes model: ", call_option_price(S0, K, T, rf, sigma))
    print("put option price according to Black-Scholes model: ", put_option_price(S0, K, T, rf, sigma))