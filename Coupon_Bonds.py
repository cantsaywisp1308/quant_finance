class CouponBonds:

    def __init__(self, principal, rate, maturity, interest_rate):
        self.principal = principal
        self.rate = rate / 100
        self.maturity = maturity
        self.interest_rate = interest_rate / 100

    def present_price(self, x, n):
        return x / (1 + self.interest_rate)**n

    def calculate_price(self):
        price = 0

        for t in range(1, self.maturity):
            price += self.present_price(self.principal * self.rate, t)

        price += self.present_price(self.principal + self.principal*self.rate, self.maturity)

        return price

if __name__ == '__main__':
    bond = CouponBonds(1000, 10, 3, 4)
    print(bond.calculate_price())