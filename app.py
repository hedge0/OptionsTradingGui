import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from scipy.optimize import minimize

class DataGenerator:
    def __init__(self, at_the_money_vol=0.2, skew=1.5, kurtosis=0.8):
        self.at_the_money_vol = at_the_money_vol
        self.skew = skew
        self.kurtosis = kurtosis
        self.data = self.generate_smile_data()

    def generate_smile_data(self):
        # Generate 40 data points instead of 20
        x = np.linspace(0.6, 1.4, 40)
        y_mid = self.at_the_money_vol + self.skew * (x - 1) ** 2 - self.kurtosis * (x - 1) ** 4
        y_mid += np.random.normal(0, 0.015, size=x.shape)
        
        # Randomly generate spread values
        spread_factors = np.random.uniform(0.01, 0.05, size=x.shape)  # Adjust min and max values as needed
        y_bid = y_mid - spread_factors
        y_ask = y_mid + spread_factors
        
        # Ensure bid prices are below mid and ask prices are above mid
        y_bid = np.maximum(y_bid, 0.0)  # Minimum volatility constraint
        y_ask = np.minimum(y_ask, 1.0)  # Maximum volatility constraint
        
        return y_mid, y_bid, y_ask, x

    def update_data(self):
        self.data = self.generate_smile_data()
        print("Data updated:", self.data)

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Implied Volatility Smile Simulation")
        self.figure, self.ax = plt.subplots(figsize=(8, 6))  # Increased figure size for clarity
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.data_gen = DataGenerator()
        self.style_plot()
        self.update_plot()
        self.update_data_and_plot()

    def svi_model(self, k, params):
        a, b, rho, m, sigma = params
        return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

    def objective_function(self, params, k, y):
        return np.sum((self.svi_model(k, params) - y) ** 2)

    def fit_svi(self, x, y):
        k = np.log(x)
        initial_guess = [0.01, 0.5, -0.3, 0.0, 0.2]
        bounds = [(0, 1), (0, 1), (-1, 1), (-1, 1), (0.01, 1)]
        result = minimize(self.objective_function, initial_guess, args=(k, y), method='L-BFGS-B', bounds=bounds)
        if not result.success:
            print("Optimization failed:", result.message)
        return result.x

    def style_plot(self):
        # Set the background color to dark
        self.ax.set_facecolor('#1c1c1c')
        self.figure.patch.set_facecolor('#1c1c1c')
        
        # Set the grid lines to be less prominent
        self.ax.grid(True, color='#444444')
        
        # Set the tick and label colors to white
        self.ax.tick_params(colors='white')
        self.ax.yaxis.label.set_color('white')
        self.ax.xaxis.label.set_color('white')
        self.ax.title.set_color('white')

    def update_plot(self):
        self.ax.clear()
        self.style_plot()  # Re-apply the styling
        
        y_mid, y_bid, y_ask, x = self.data_gen.data
        
        # Plot bid, ask, and midpoints, with a larger size for the midpoint
        for i in range(len(x)):
            self.ax.plot([x[i], x[i]], [y_bid[i], y_ask[i]], color='red', linewidth=0.5)  # Vertical line
            self.ax.scatter([x[i]], [y_mid[i]], color='red', s=20)  # Larger Midpoint
            self.ax.scatter([x[i]], [y_bid[i]], color='red', s=10)  # Bid
            self.ax.scatter([x[i]], [y_ask[i]], color='red', s=10)  # Ask
        
        params = self.fit_svi(x, y_mid)
        fine_x = np.linspace(min(x), max(x), 200)  # Increased number of points for a smoother fit line
        fine_k = np.log(fine_x)
        interpolated_y = self.svi_model(fine_k, params)
        self.ax.plot(fine_x, interpolated_y, color='green', label="SVI Fit", linewidth=1.5)
        
        # Set a fixed y-axis range
        self.ax.set_ylim(0.0, 0.75)  # Adjust these values as necessary to accommodate your data
        
        self.ax.set_title("Implied Volatility Smile")
        self.ax.set_xlabel("Moneyness = Strike / Forward Price")
        self.ax.set_ylabel("Implied Volatility")
        self.ax.legend(loc='upper left', fontsize='small', facecolor='green', edgecolor='white')
        self.canvas.draw()

    def update_data_and_plot(self):
        self.data_gen.update_data()
        self.update_plot()
        self.root.after(5000, self.update_data_and_plot)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
