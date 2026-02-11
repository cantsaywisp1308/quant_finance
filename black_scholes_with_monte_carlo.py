import numpy as np

class OptionPricing:

    def __init__(self, S0, E, T, rf, sigma, iterations):
        self.S0 = S0
        self.E = E
        self.T = T
        self.rf = rf
        self.sigma = sigma
        self.iterations = iterations

    def call_option_pricing(self):
        # we have 2 columns first with 0s and second column is the payoff
        # we need the first column to be 0s: payoff function is max (0, S-E) for call option
        option_data = np.zeros([self.iterations, 2])

        #dimensions: 1 dimensional array with as many items as the iterations
        rand = np.random.normal(0, 1, [1, self.iterations])
        stock_price = self.S0 * np.exp(self.T * (self.rf - 0.5 * self.sigma ** 2) + self.sigma * np.sqrt(self.T) * rand)

        #we calculate the S-E for option
        option_data[:, 1] = stock_price - self.E

        #we use average for Monte Carlo simulation
        #max() returns the max(0, S-E) for call options
        #this is the average value
        average = np.sum(np.amax(option_data, 1)) / float(self.iterations)

        return np.exp(-1.0 * self.rf * self.T) * average

    def put_option_pricing(self):
        # we have 2 columns first with 0s and second column is the payoff
        # we need the first column to be 0s: payoff function is max (0, S-E) for call option
        option_data = np.zeros([self.iterations, 2])

        #dimensions: 1 dimensional array with as many items as the iterations
        rand = np.random.normal(0, 1, [1, self.iterations])
        stock_price = self.S0 * np.exp(self.T * (self.rf - 0.5 * self.sigma ** 2) + self.sigma * np.sqrt(self.T) * rand)

        #we calculate the E-S for option
        option_data[:, 1] = self.E - stock_price

        #we use average for Monte Carlo simulation
        #max() returns the max(0, S-E) for call options
        #this is the average value
        average = np.sum(np.amax(option_data, 1)) / float(self.iterations)

        return np.exp(-1.0 * self.rf * self.T) * average

if __name__ == '__main__':
    model = OptionPricing(306.04, 310, 2, 0.046, 0.3431,1000)
    print('Value of the call option is $%.2f ' %model.call_option_pricing())
    print('Value of the put option is $%.2f ' % model.put_option_pricing())
    # model.call_option_pricing()