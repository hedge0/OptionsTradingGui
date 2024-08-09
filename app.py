import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from scipy.optimize import minimize
from tastytrade import Session
from tastytrade.utils import TastytradeError
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Global session and config
session = None
config = {}

def load_config():
    global config
    config = {
        "TASTYTRADE_USERNAME": os.getenv('TASTYTRADE_USERNAME'),
        "TASTYTRADE_PASSWORD": os.getenv('TASTYTRADE_PASSWORD'),
        "TASTYTRADE_ACCOUNT_NUMBER": os.getenv('TASTYTRADE_ACCOUNT_NUMBER'),
    }

    for key, value in config.items():
        if value is None:
            raise ValueError(f"{key} environment variable not set")

class DataGenerator:
    def __init__(self, at_the_money_vol=0.2, skew=1.5, kurtosis=0.8):
        self.at_the_money_vol = at_the_money_vol
        self.skew = skew
        self.kurtosis = kurtosis
        self.data = self.generate_smile_data()

    def generate_smile_data(self):
        x = np.linspace(0.6, 1.4, 40)
        y_mid = self.at_the_money_vol + self.skew * (x - 1) ** 2 - self.kurtosis * (x - 1) ** 4
        y_mid += np.random.normal(0, 0.015, size=x.shape)
        spread_factors = np.random.uniform(0.01, 0.05, size=x.shape)
        y_bid = np.maximum(y_mid - spread_factors, 0.0)
        y_ask = np.minimum(y_mid + spread_factors, 1.0)
        return y_mid, y_bid, y_ask, x

    def update_data(self):
        self.data = self.generate_smile_data()

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Implied Volatility Smile Simulation")
        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.data_gen = DataGenerator()
        self.fine_x = np.linspace(0.6, 1.4, 200)
        self.fine_k = np.log(self.fine_x)
        self.setup_plot()
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
        return result.x

    def setup_plot(self):
        self.ax.set_facecolor('#1c1c1c')
        self.figure.patch.set_facecolor('#1c1c1c')
        self.ax.grid(True, color='#444444')
        self.ax.tick_params(colors='white')
        self.ax.yaxis.label.set_color('white')
        self.ax.xaxis.label.set_color('white')
        self.ax.title.set_color('white')
        self.ax.set_ylim(0.0, 0.75)
        self.ax.set_title("Implied Volatility Smile")
        self.ax.set_xlabel("Strike K")
        self.ax.set_ylabel("Implied Volatility")

    def update_plot(self):
        y_mid, y_bid, y_ask, x = self.data_gen.data

        if hasattr(self, 'midpoints'):
            self.midpoints.set_offsets(np.c_[x, y_mid])
            self.bids.set_offsets(np.c_[x, y_bid])
            self.asks.set_offsets(np.c_[x, y_ask])
            
            for i, line in enumerate(self.lines):
                line.set_data([x[i], x[i]], [y_bid[i], y_ask[i]])
        else:
            self.bids = self.ax.scatter(x, y_bid, color='red', s=10, label="Bid")
            self.asks = self.ax.scatter(x, y_ask, color='red', s=10, label="Ask")
            self.midpoints = self.ax.scatter(x, y_mid, color='red', s=20, label="Midpoint")
            self.lines = [self.ax.plot([x[i], x[i]], [y_bid[i], y_ask[i]], color='red', linewidth=0.5)[0] for i in range(len(x))]

        params = self.fit_svi(x, y_mid)
        interpolated_y = self.svi_model(self.fine_k, params)

        if hasattr(self, 'fit_line'):
            self.fit_line.set_data(self.fine_x, interpolated_y)
        else:
            self.fit_line, = self.ax.plot(self.fine_x, interpolated_y, color='green', label="SVI Fit", linewidth=1.5)

        self.canvas.draw()

    def update_data_and_plot(self):
        self.data_gen.update_data()
        self.update_plot()
        self.root.after(5000, self.update_data_and_plot)

def show_login():
    login_window = tk.Tk()
    login_window.title("Login")
    login_window.geometry("300x200")

    tk.Label(login_window, text="Username:").pack(pady=5)
    username_entry = tk.Entry(login_window)
    username_entry.insert(0, config['TASTYTRADE_USERNAME'])
    username_entry.pack(pady=5)

    tk.Label(login_window, text="Password:").pack(pady=5)
    password_entry = tk.Entry(login_window, show="*")
    password_entry.insert(0, config['TASTYTRADE_PASSWORD'])
    password_entry.pack(pady=5)

    def check_credentials():
        global session  # Declare session as global
        username = username_entry.get()
        password = password_entry.get()
        try:
            session = Session(login=username, password=password, remember_me=True)
            messagebox.showinfo("Login Success", "Login successful!")
            login_window.destroy()
            open_main_app()
        except TastytradeError as e:
            if "invalid_credentials" in str(e):
                messagebox.showerror("Login Failed", "Incorrect Username or Password")
            else:
                messagebox.showerror("Login Failed", f"An error occurred: {str(e)}")

    tk.Button(login_window, text="Login", command=check_credentials).pack(pady=20)

    login_window.mainloop()

def open_main_app():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    load_config()
    show_login()
