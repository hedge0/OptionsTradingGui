from collections import defaultdict

class DataGenerator:
    """
    A class to generate synthetic implied volatility smile data or return predefined data.

    Attributes:
        at_the_money_vol (float): The at-the-money volatility.
        skew (float): The skewness of the volatility smile.
        kurtosis (float): The kurtosis of the volatility smile.
        asymmetry (float): The asymmetry of the volatility smile.
        tail_rise (float): The tail rise in the volatility smile.
        data (defaultdict): Generated data containing bid, ask, and mid prices mapped to strikes.
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
        self.reset_mid_to_average()

    def generate_smile_data(self):
        """
        Return the predefined implied volatility smile data.

        Returns:
            defaultdict: Containing the bid, ask, and mid prices mapped to strikes.
        """
        predefined_data = {
            95.0: {'bid': 81.95, 'ask': 82.4, 'mid': 0.0},
            100.0: {'bid': 76.95, 'ask': 77.4, 'mid': 0.0},
            105.0: {'bid': 72.0, 'ask': 72.4, 'mid': 0.0},
            110.0: {'bid': 67.0, 'ask': 67.35, 'mid': 0.0},
            115.0: {'bid': 62.0, 'ask': 62.45, 'mid': 0.0},
            120.0: {'bid': 57.0, 'ask': 57.45, 'mid': 0.0},
            125.0: {'bid': 52.0, 'ask': 52.35, 'mid': 0.0},
            130.0: {'bid': 47.0, 'ask': 47.5, 'mid': 0.0},
            135.0: {'bid': 42.05, 'ask': 42.5, 'mid': 0.0},
            140.0: {'bid': 37.05, 'ask': 37.45, 'mid': 0.0},
            145.0: {'bid': 32.1, 'ask': 32.5, 'mid': 0.0},
            150.0: {'bid': 27.15, 'ask': 27.55, 'mid': 0.0},
            152.5: {'bid': 24.75, 'ask': 25.1, 'mid': 0.0},
            155.0: {'bid': 22.3, 'ask': 22.55, 'mid': 0.0},
            157.5: {'bid': 19.85, 'ask': 20.2, 'mid': 0.0},
            160.0: {'bid': 17.4, 'ask': 17.7, 'mid': 0.0},
            162.5: {'bid': 14.95, 'ask': 15.2, 'mid': 0.0},
            165.0: {'bid': 12.55, 'ask': 12.75, 'mid': 0.0},
            167.5: {'bid': 10.25, 'ask': 10.4, 'mid': 0.0},
            170.0: {'bid': 8.05, 'ask': 8.2, 'mid': 0.0},
            172.5: {'bid': 6.05, 'ask': 6.2, 'mid': 0.0},
            175.0: {'bid': 4.35, 'ask': 4.45, 'mid': 0.0},
            177.5: {'bid': 2.97, 'ask': 3.05, 'mid': 0.0},
            180.0: {'bid': 1.91, 'ask': 1.95, 'mid': 0.0},
            182.5: {'bid': 1.18, 'ask': 1.2, 'mid': 0.0},
            185.0: {'bid': 0.7, 'ask': 0.72, 'mid': 0.0},
            187.5: {'bid': 0.42, 'ask': 0.44, 'mid': 0.0},
            190.0: {'bid': 0.25, 'ask': 0.27, 'mid': 0.0},
            192.5: {'bid': 0.16, 'ask': 0.18, 'mid': 0.0},
            195.0: {'bid': 0.11, 'ask': 0.12, 'mid': 0.0},
            197.5: {'bid': 0.08, 'ask': 0.09, 'mid': 0.0},
            200.0: {'bid': 0.07, 'ask': 0.08, 'mid': 0.0},
            202.5: {'bid': 0.05, 'ask': 0.06, 'mid': 0.0},
            205.0: {'bid': 0.04, 'ask': 0.05, 'mid': 0.0},
            210.0: {'bid': 0.03, 'ask': 0.04, 'mid': 0.0},
            215.0: {'bid': 0.02, 'ask': 0.03, 'mid': 0.0},
            220.0: {'bid': 0.02, 'ask': 0.03, 'mid': 0.0},
            225.0: {'bid': 0.01, 'ask': 0.02, 'mid': 0.0},
            230.0: {'bid': 0.01, 'ask': 0.02, 'mid': 0.0},
            235.0: {'bid': 0.0, 'ask': 0.01, 'mid': 0.0},
            240.0: {'bid': 0.0, 'ask': 0.01, 'mid': 0.0},
            245.0: {'bid': 0.0, 'ask': 0.01, 'mid': 0.0},
            250.0: {'bid': 0.0, 'ask': 0.01, 'mid': 0.0},
            255.0: {'bid': 0.0, 'ask': 0.01, 'mid': 0.0},
            260.0: {'bid': 0.0, 'ask': 0.01, 'mid': 0.0},
            265.0: {'bid': 0.0, 'ask': 0.01, 'mid': 0.0},
            270.0: {'bid': 0.0, 'ask': 0.01, 'mid': 0.0}
        }
        
        quote_data = defaultdict(lambda: {"bid": None, "ask": None, "mid": None})
        for strike, prices in predefined_data.items():
            quote_data[strike] = prices
        
        return quote_data

    def reset_mid_to_average(self):
        """
        Reset the mid prices to the average of bid and ask prices.
        """
        for strike, prices in self.data.items():
            prices['mid'] = (prices['bid'] + prices['ask']) / 2

    def update_data(self):
        """
        Update the generated volatility smile data with the predefined data
        and reset the mid prices.
        """
        self.data = self.generate_smile_data()
        self.reset_mid_to_average()
