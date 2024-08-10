import tkinter as tk
from tkinter import messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from scipy.optimize import minimize
from tastytrade import Session, Account
from tastytrade.utils import TastytradeError
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

config = {}
session = None
account = None

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
        x = np.linspace(0.6, 1.4, 80)
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
        self.selected_method = tk.StringVar(value="RFV")

        style = ttk.Style()
        style.theme_use('clam')

        style.configure("TCombobox",
                        fieldbackground="white", background="white",
                        selectbackground="white", selectforeground="black")

        style.map('TCombobox',
                  background=[('readonly', 'white')],
                  fieldbackground=[('readonly', 'white')],
                  foreground=[('readonly', 'black')],
                  selectbackground=[('readonly', 'white')],
                  selectforeground=[('readonly', 'black')])

        self.create_dropdown_and_button()
        self.setup_plot()
        self.update_plot()
        self.update_data_and_plot()

    def create_dropdown_and_button(self):
        dropdown_frame = tk.Frame(self.root)
        dropdown_frame.pack(side=tk.TOP, anchor=tk.NE, padx=10, pady=10)
        
        # Metrics display frame (placed first on the left)
        metrics_frame = tk.Frame(dropdown_frame)
        metrics_frame.pack(side=tk.LEFT, padx=5)

        # Adjusted metrics label styling
        self.metrics_text = tk.Label(metrics_frame, text="χ²: N/A    avE5: N/A bps", fg='black', bg=dropdown_frame.cget('background'))
        self.metrics_text.pack(side=tk.LEFT)
        
        # Create a frame for the model selection and metrics
        selection_and_metrics_frame = tk.Frame(dropdown_frame)
        selection_and_metrics_frame.pack(side=tk.LEFT)

        tk.Label(selection_and_metrics_frame, text="Select Model:").pack(side=tk.LEFT)
        self.method_menu = ttk.Combobox(selection_and_metrics_frame, textvariable=self.selected_method, 
                                        values=["RFV", "SVI", "SLV", "SABR"], state="readonly", style="TCombobox")
        self.method_menu.pack(side=tk.LEFT, padx=5)
        
        tk.Button(selection_and_metrics_frame, text="Enter", command=self.update_plot).pack(side=tk.LEFT)


    def svi_model(self, k, params):
        a, b, rho, m, sigma = params
        return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

    def slv_model(self, k, params):
        a, b, c, d, e = params
        return a + b * k + c * k**2 + d * k**3 + e * k**4

    def rfv_model(self, k, params):
        a, b, c, d, e = params
        return (a + b*k + c*k**2) / (1 + d*k + e*k**2)

    def sabr_model(self, k, params):
        alpha, beta, rho, nu, f0 = params
        return alpha * (1 + beta * k + rho * k**2 + nu * k**3 + f0 * k**4)

    def objective_function(self, params, k, y, model):
        return np.sum((model(k, params) - y) ** 2)

    def fit_model(self, x, y, model):
        k = np.log(x)
        if model == self.svi_model:
            initial_guess = [0.01, 0.5, -0.3, 0.0, 0.2]
            bounds = [(0, 1), (0, 1), (-1, 1), (-1, 1), (0.01, 1)]
        else:
            initial_guess = [0.2, 0.3, 0.1, 0.2, 0.1]
            bounds = [(None, None), (None, None), (None, None), (None, None), (None, None)]
        result = minimize(self.objective_function, initial_guess, args=(k, y, model), method='L-BFGS-B', bounds=bounds)
        return result.x

    def compute_metrics(self, x, y, model, params):
        k = np.log(x)
        y_fit = model(k, params)
        
        # Chi-Squared Calculation
        chi_squared = np.sum((y - y_fit) ** 2)
        
        # Average Error (avE5) Calculation
        avE5 = np.mean(np.abs(y - y_fit)) * 10000
        
        return chi_squared, avE5

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

        model = {
            "SVI": self.svi_model,
            "SLV": self.slv_model,
            "RFV": self.rfv_model,
            "SABR": self.sabr_model
        }.get(self.selected_method.get())

        params = self.fit_model(x, y_mid, model)
        interpolated_y = model(np.log(self.fine_x), params)

        if hasattr(self, 'fit_line'):
            self.fit_line.set_data(self.fine_x, interpolated_y)
        else:
            self.fit_line, = self.ax.plot(self.fine_x, interpolated_y, color='green', label="Fit", linewidth=1.5)
        
        # Compute and display metrics
        chi_squared, avE5 = self.compute_metrics(x, y_mid, model, params)
        
        self.metrics_text.config(text=f"χ²: {chi_squared:.4f}    avE5: {avE5:.2f} bps")

        self.canvas.draw()

    def update_data_and_plot(self):
        self.data_gen.update_data()
        self.update_plot()
        self.root.after(15000, self.update_data_and_plot)  # 15 seconds interval

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
        global session
        username = username_entry.get()
        password = password_entry.get()
        try:
            session = Session(login=username, password=password, remember_me=True)
            messagebox.showinfo("Login Success", "Login successful!")

            for widget in login_window.winfo_children():
                widget.pack_forget()
            
            tk.Label(login_window, text="Account Number:").pack(pady=5)
            account_entry = tk.Entry(login_window)
            account_entry.insert(0, config['TASTYTRADE_ACCOUNT_NUMBER'])
            account_entry.pack(pady=5)

            tk.Label(login_window, text="Ticker:").pack(pady=5)
            ticker_entry = tk.Entry(login_window)
            ticker_entry.pack(pady=5)

            def validate_account_and_open_plot():
                global account
                account_number = account_entry.get()
                try:
                    account = Account.get_account(session, account_number)
                    messagebox.showinfo("Account Validated", "Account number validated successfully!")
                    login_window.destroy()
                    open_main_app()
                except TastytradeError as e:
                    if "record_not_found" in str(e):
                        messagebox.showerror("Validation Failed", f"Invalid account number: {account_number}. Please check and try again.")
                    else:
                        messagebox.showerror("Validation Failed", f"An error occurred: {str(e)}")

            tk.Button(login_window, text="Enter", command=validate_account_and_open_plot).pack(pady=20)

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
