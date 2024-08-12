import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tastytrade import Session
from tastytrade.utils import TastytradeError
from tastytrade.instruments import NestedOptionChain
from dotenv import load_dotenv
import os
from models import svi_model, slv_model, rfv_model, sabr_model, fit_model, compute_metrics
from data_generator import DataGenerator

# Load environment variables
load_dotenv()

config = {}
session = None
chain = None
expiration_to_strikes_map = {}
streamer_to_strike_map = {}
expiration_dates_list = []

def load_config():
    global config
    config = {
        "TASTYTRADE_USERNAME": os.getenv('TASTYTRADE_USERNAME'),
        "TASTYTRADE_PASSWORD": os.getenv('TASTYTRADE_PASSWORD'),
    }

    for key, value in config.items():
        if value is None:
            raise ValueError(f"{key} environment variable not set")

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
        global session, chain
        username = username_entry.get()
        password = password_entry.get()
        try:
            session = Session(login=username, password=password, remember_me=True)
            messagebox.showinfo("Login Success", "Login successful!")

            for widget in login_window.winfo_children():
                widget.pack_forget()
            
            tk.Label(login_window, text="Ticker:").pack(pady=5)
            ticker_entry = tk.Entry(login_window)
            ticker_entry.insert(0, "SPY")  # Set default value to "SPY"
            ticker_entry.pack(pady=5)

            def validate_ticker_and_open_plot():
                global chain, expiration_to_strikes_map, streamer_to_strike_map, expiration_dates_list
                ticker = ticker_entry.get()  # Get the entered ticker value

                try:
                    chain = NestedOptionChain.get_chain(session, ticker)
                    messagebox.showinfo("Ticker Validated", "Ticker validated successfully!")

                    # Initialize expiration_to_strikes_map and expiration_dates_list
                    if chain is not None:
                        expiration_to_strikes_map = {}
                        streamer_to_strike_map = {}
                        expiration_dates_list = []

                        for expiration in chain.expirations:
                            calls_list = []
                            puts_list = []

                            for strike in expiration.strikes:
                                # Populate the calls and puts lists
                                calls_list.append(strike.call_streamer_symbol)
                                puts_list.append(strike.put_streamer_symbol)

                                # Populate the streamer_to_strike_map
                                streamer_to_strike_map[strike.call_streamer_symbol] = strike.strike_price
                                streamer_to_strike_map[strike.put_streamer_symbol] = strike.strike_price

                            # Map each expiration date to the calls and puts lists
                            expiration_to_strikes_map[expiration.expiration_date] = {
                                "calls": calls_list,
                                "puts": puts_list
                            }
                            # Add the expiration date to the list
                            expiration_dates_list.append(expiration.expiration_date)

                except TastytradeError as e:
                    if "record_not_found" in str(e):
                        messagebox.showerror("Validation Failed", f"Invalid ticker symbol: {ticker}. Please check and try again.")
                        return
                    else:
                        messagebox.showerror("Validation Failed", f"An error occurred: {str(e)}")
                        return

                login_window.destroy()
                open_main_app(ticker)  # Pass the ticker to the main app

            tk.Button(login_window, text="Enter", command=validate_ticker_and_open_plot).pack(pady=20)

        except TastytradeError as e:
            if "invalid_credentials" in str(e):
                messagebox.showerror("Login Failed", "Incorrect Username or Password")
            else:
                messagebox.showerror("Login Failed", f"An error occurred: {str(e)}")

    tk.Button(login_window, text="Login", command=check_credentials).pack(pady=20)

    login_window.mainloop()

def open_main_app(ticker):
    root = tk.Tk()
    app = App(root, ticker)  # Pass the ticker to the App class
    root.mainloop()

class App:
    def __init__(self, root, ticker):
        self.root = root
        self.root.title("Implied Volatility Smile Simulation")
        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.data_gen = DataGenerator()
        self.fine_x = np.linspace(0.6, 1.4, 200)
        self.selected_method = tk.StringVar(value="RFV")
        self.selected_objective = tk.StringVar(value="WRE")
        self.ticker = ticker  # Store the ticker value

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

        tk.Label(selection_and_metrics_frame, text="Model:").pack(side=tk.LEFT)
        self.method_menu = ttk.Combobox(selection_and_metrics_frame, textvariable=self.selected_method, 
                                        values=["RFV", "SVI", "SLV", "SABR"], state="readonly", style="TCombobox")
        self.method_menu.pack(side=tk.LEFT, padx=5)

        tk.Label(selection_and_metrics_frame, text="Objective Function:").pack(side=tk.LEFT, padx=5)
        self.objective_menu = ttk.Combobox(selection_and_metrics_frame, textvariable=self.selected_objective, 
                                           values=["WRE", "WLS", "LS", "RE"], state="readonly", style="TCombobox")
        self.objective_menu.pack(side=tk.LEFT, padx=5)

        # Add a filter input field
        tk.Label(selection_and_metrics_frame, text="Max Spread:").pack(side=tk.LEFT, padx=5)
        self.spread_filter_var = tk.StringVar(value="0.0")
        self.spread_filter_entry = tk.Entry(selection_and_metrics_frame, textvariable=self.spread_filter_var, width=10)
        self.spread_filter_entry.pack(side=tk.LEFT, padx=5)

        # Add the Exp. Date dropdown menu
        tk.Label(selection_and_metrics_frame, text="Exp. Date:").pack(side=tk.LEFT, padx=5)
        self.exp_date_var = tk.StringVar(value=expiration_dates_list[0])  # Default to the first date
        self.exp_date_menu = ttk.Combobox(selection_and_metrics_frame, textvariable=self.exp_date_var, 
                                          values=expiration_dates_list, state="readonly", style="TCombobox")
        self.exp_date_menu.pack(side=tk.LEFT, padx=5)

        # Add the Type dropdown menu
        tk.Label(selection_and_metrics_frame, text="Type:").pack(side=tk.LEFT, padx=5)
        self.type_var = tk.StringVar(value="calls")  # Default to "calls"
        self.type_menu = ttk.Combobox(selection_and_metrics_frame, textvariable=self.type_var, 
                                      values=["calls", "puts"], state="readonly", style="TCombobox")
        self.type_menu.pack(side=tk.LEFT, padx=5)

        tk.Button(selection_and_metrics_frame, text="Enter", command=self.update_plot).pack(side=tk.LEFT)

    def setup_plot(self):
        self.ax.set_facecolor('#1c1c1c')
        self.figure.patch.set_facecolor('#1c1c1c')
        self.ax.grid(True, color='#444444')
        self.ax.tick_params(colors='white')
        self.ax.yaxis.label.set_color('white')
        self.ax.xaxis.label.set_color('white')
        self.ax.title.set_color('white')
        self.ax.set_ylim(0.0, 0.75)
        self.ax.set_title(f"{self.ticker}")
        self.ax.set_xlabel("Strike K")
        self.ax.set_ylabel("Implied Volatility")

    def update_plot(self):
        y_mid, y_bid, y_ask, x = self.data_gen.data

        # Apply the bid-ask spread filter if necessary
        try:
            max_spread = float(self.spread_filter_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number for Max Bid-Ask Spread.")
            return
        
        if (max_spread > 0.0):
            mask = (y_ask - y_bid) <= max_spread
            x = x[mask]
            y_mid = y_mid[mask]
            y_bid = y_bid[mask]
            y_ask = y_ask[mask]

        # Check if any data points remain after filtering
        if len(x) == 0:
            messagebox.showwarning("No Data", "All data points were filtered out. Adjust the spread filter.")
            return

        # Clear previous scatter plots if they exist
        if hasattr(self, 'midpoints'):
            self.midpoints.remove()
            self.bids.remove()
            self.asks.remove()
            for line in self.lines:
                line.remove()

        self.bids = self.ax.scatter(x, y_bid, color='red', s=10, label="Bid")
        self.asks = self.ax.scatter(x, y_ask, color='red', s=10, label="Ask")
        self.midpoints = self.ax.scatter(x, y_mid, color='red', s=20, label="Midpoint")
        self.lines = [self.ax.plot([x[i], x[i]], [y_bid[i], y_ask[i]], color='red', linewidth=0.5)[0] for i in range(len(x))]

        model = {
            "SVI": svi_model,
            "SLV": slv_model,
            "RFV": rfv_model,
            "SABR": sabr_model,
        }.get(self.selected_method.get())

        # Apply the selected objective function
        params = fit_model(x, y_mid, y_bid, y_ask, model, method=self.selected_objective.get())

        interpolated_y = model(np.log(self.fine_x), params)

        if hasattr(self, 'fit_line'):
            self.fit_line.set_data(self.fine_x, interpolated_y)
        else:
            self.fit_line, = self.ax.plot(self.fine_x, interpolated_y, color='green', label="Fit", linewidth=1.5)
        
        # Compute and display metrics
        chi_squared, avE5 = compute_metrics(x, y_mid, model, params)
        
        self.metrics_text.config(text=f"χ²: {chi_squared:.4f}    avE5: {avE5:.2f} bps")

        self.canvas.draw()

    def update_data_and_plot(self):
        self.data_gen.update_data()
        self.update_plot()
        self.root.after(10000, self.update_data_and_plot)  # 10 seconds interval

if __name__ == "__main__":
    load_config()
    show_login()