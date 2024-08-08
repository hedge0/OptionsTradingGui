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
        x = np.linspace(0.6, 1.4, 20)
        y = self.at_the_money_vol + self.skew * (x - 1) ** 2 - self.kurtosis * (x - 1) ** 4
        y += np.random.normal(0, 0.015, size=x.shape)
        tail_factor = 1.5
        y = np.where(x < 1, y * tail_factor, y * tail_factor)
        return y, x

    def update_data(self):
        self.data = self.generate_smile_data()
        print("Data updated:", self.data)

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Implied Volatility Smile Simulation")
        self.figure, self.ax = plt.subplots(figsize=(6, 5))
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.data_gen = DataGenerator()
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

    def update_plot(self):
        self.ax.clear()
        y, x = self.data_gen.data
        self.ax.scatter(x, y, marker='o', label="Raw Data")
        params = self.fit_svi(x, y)
        fine_x = np.linspace(min(x), max(x), 100)
        fine_k = np.log(fine_x)
        interpolated_y = self.svi_model(fine_k, params)
        self.ax.plot(fine_x, interpolated_y, color='red', label="SVI Model Fit")
        self.ax.set_ylim(0.2, 1.0)
        self.ax.set_title("Implied Volatility Smile")
        self.ax.set_xlabel("Moneyness = Strike / Forward Price")
        self.ax.set_ylabel("Implied Volatility")
        self.ax.legend()
        self.canvas.draw()

    def update_data_and_plot(self):
        self.data_gen.update_data()
        self.update_plot()
        self.root.after(5000, self.update_data_and_plot)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()