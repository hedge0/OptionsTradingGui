import numpy as np

class DataGenerator:
    """
    A class to generate synthetic implied volatility smile data.

    Attributes:
        at_the_money_vol (float): The at-the-money volatility.
        skew (float): The skewness of the volatility smile.
        kurtosis (float): The kurtosis of the volatility smile.
        asymmetry (float): The asymmetry of the volatility smile.
        tail_rise (float): The tail rise in the volatility smile.
        data (tuple): Generated data containing mid prices, bid prices, ask prices, and strikes.
    """
    
    def __init__(self, at_the_money_vol=0.2, skew=1.0, kurtosis=0.6, asymmetry=0.1, tail_rise=0.3):
        """
        Initialize the DataGenerator with specific characteristics of the volatility smile.

        Args:
            at_the_money_vol (float, optional): The at-the-money volatility. Defaults to 0.2.
            skew (float, optional): The skewness of the volatility smile. Defaults to 1.0.
            kurtosis (float, optional): The kurtosis of the volatility smile. Defaults to 0.6.
            asymmetry (float, optional): The asymmetry of the volatility smile. Defaults to 0.1.
            tail_rise (float, optional): The tail rise in the volatility smile. Defaults to 0.3.
        """
        self.at_the_money_vol = at_the_money_vol
        self.skew = skew
        self.kurtosis = kurtosis
        self.asymmetry = asymmetry
        self.tail_rise = tail_rise
        self.data = self.generate_smile_data()

    def generate_smile_data(self):
        """
        Generate synthetic implied volatility smile data.

        Returns:
            tuple: Containing the mid prices (y_mid), bid prices (y_bid), ask prices (y_ask), and strikes (x).
        """
        x = np.linspace(0.6, 1.4, 80)
        
        y_mid = (self.at_the_money_vol + 
                 self.skew * (x - 1) ** 2 - 
                 self.kurtosis * (x - 1) ** 4 +
                 self.asymmetry * (x - 1) +
                 self.tail_rise * (x - 1) ** 3)
        
        y_mid += np.random.normal(0, 0.01, size=x.shape)
        
        spread_factors = np.random.uniform(0.005, 0.03, size=x.shape)
        
        tail_end_mask = x > 1.2
        random_wider_spreads = np.random.rand(len(x)) > 0.8
        spread_factors += tail_end_mask * random_wider_spreads * np.random.uniform(0.03, 0.07, size=x.shape)
        
        y_bid = np.maximum(y_mid - spread_factors, 0.0)
        y_ask = np.minimum(y_mid + spread_factors, 1.0)
        
        return y_mid, y_bid, y_ask, x

    def update_data(self):
        """
        Update the generated volatility smile data with new synthetic data.
        """
        self.data = self.generate_smile_data()
