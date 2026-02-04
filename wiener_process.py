import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy.random as npr
def wiener_process(dt = 0.1, x0 = 0, n = 1000): #also called brownian process
    #W(t=0) = 0
    #initialize W(t) with zeros
    w = np.zeros(n+1)

    #we create N+1 timestamps  t = 0 ... N
    t = np.linspace(x0, n, n+1)

    #we use cumulative sum: on every step of the additional value is drawn from  a normal distribution
    #with mean = 0 and variance dt ... N(0, t)
    #by the way N(0, t) = sqrt(dt) * N(0, 1) usually this formula is used

    w[1:n+1] = np.cumsum(np.random.normal(0, np.sqrt(dt), n))
    return t, w

def plot_process(t, w):
    plt.plot(t, w)
    plt.xlabel('Time(t)')
    plt.ylabel('Wiener process W(t)')
    plt.title('Wiener process W(t)')
    plt.show()

if __name__ == '__main__':
    time, data = wiener_process()
    plot_process(time, data)