import numpy as np

class DataGenerator:
    def __init__(self, at_the_money_vol=0.2, skew=1.0, kurtosis=0.6, asymmetry=0.1, tail_rise=0.3):
        self.at_the_money_vol = at_the_money_vol
        self.skew = skew
        self.kurtosis = kurtosis
        self.asymmetry = asymmetry
        self.tail_rise = tail_rise
        self.data = self.generate_smile_data()

    def generate_smile_data(self):
        x = np.linspace(0.6, 1.4, 80)
        
        # Generate a more aggressive rise in the right tail
        y_mid = (self.at_the_money_vol + 
                 self.skew * (x - 1) ** 2 - 
                 self.kurtosis * (x - 1) ** 4 +
                 self.asymmetry * (x - 1) +
                 self.tail_rise * (x - 1) ** 3)  # Added cubic term for tail rise
        
        # Reduce the noise for smoother curve
        y_mid += np.random.normal(0, 0.01, size=x.shape)
        
        # Spread factors are slightly tighter to reflect more liquid market conditions
        spread_factors = np.random.uniform(0.005, 0.03, size=x.shape)
        
        # Introduce wider spreads for some tail-end options
        tail_end_mask = x > 1.2  # Focus on the right tail (deep OTM options)
        random_wider_spreads = np.random.rand(len(x)) > 0.8  # 20% chance for wider spreads in the tail
        spread_factors += tail_end_mask * random_wider_spreads * np.random.uniform(0.03, 0.07, size=x.shape)
        
        y_bid = np.maximum(y_mid - spread_factors, 0.0)
        y_ask = np.minimum(y_mid + spread_factors, 1.0)
        
        return y_mid, y_bid, y_ask, x

    def update_data(self):
        self.data = self.generate_smile_data()
