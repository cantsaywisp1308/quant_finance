import numpy as np
from scipy.stats import norm
import yfinance as yf
from datetime import datetime

def calculate_black_scholes_IBM(S, K, T, r, sigma):
    #calculate d1
    d1 = (np.log(S/K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    #calculate d2
    d2 = d1 - sigma * np.sqrt(T)

    #use scipy.stats.norm to calculate N(d1) and N(d2)
    n_d1 = norm.cdf(d1)
    n_d2 = norm.cdf(d2)

    #find call price
    call_price = (S * n_d1) - (K * np.exp(-r * T) * n_d2)

    return call_price, n_d1, n_d2

def get_stock_data():
    ibm = yf.Ticker("IBM")
    stock_price = ibm.fast_info['last_price']
    #get expiration dates and pick one (6 months for example)
    expirations = ibm.options
    select_expiry = expirations[5]

    #get option chain
    opt = ibm.option_chain(select_expiry)
    calls = opt.calls

    print(calls)

def calculate_T(selected_expiry):
    expiry_date = datetime.strptime(selected_expiry, "%Y-%m-%d")
    days_to_expiry = expiry_date - datetime.now()
    T = days_to_expiry/ 365.0

    return T

def get_risk_free_rate():
    return yf.Ticker('^IRX').fast_info['last_price'] / 100



if __name__ == '__main__':
    # call_price, n_d1, n_d2 = calculate_black_scholes_IBM(306.04, 310, 0.5, 0.0361, 0.3181)
    # print('Call price: ', call_price)
    # print('N(d1): ', n_d1)
    # print('N(d2): ', n_d2)
    #get_stock_data()
    rfr = get_risk_free_rate()
    print(rfr)