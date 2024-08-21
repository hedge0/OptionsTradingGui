import asyncio
import math
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk
from tkinter import messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tastytrade import DXLinkStreamer
from tastytrade.dxfeed import EventType
from data_generator import DataGenerator
from models import filter_strikes, svi_model, slv_model, rfv_model, sabr_model, rbf_model, fit_model, compute_metrics, calculate_implied_volatility_lr, calculate_implied_volatility_baw, barone_adesi_whaley_american_option_price, leisen_reimer_tree
from plot_interaction import on_mouse_move, on_scroll, on_press, on_release

class PlotManager:
    def __init__(self, root, ticker, session, expiration_to_strikes_map, streamer_to_strike_map, expiration_dates_list, risk_free_rate):
        self.root = root
        self.session = session
        self.expiration_to_strikes_map = expiration_to_strikes_map
        self.streamer_to_strike_map = streamer_to_strike_map
        self.expiration_dates_list = expiration_dates_list
        self.risk_free_rate = risk_free_rate / 100

        self.root.title("Implied Volatility Smile Simulation")
        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.data_gen = DataGenerator()
        self.selected_method = tk.StringVar(value="RBF")
        self.selected_objective = tk.StringVar(value="WLS")
        self.ticker = ticker
        self.selected_pricing_model = tk.StringVar(value="Leisen-Reimer")
        self.press_event = None

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
        self.canvas.mpl_connect('scroll_event', lambda event: on_scroll(event, self))
        self.canvas.mpl_connect('motion_notify_event', lambda event: on_mouse_move(event, self))
        self.canvas.mpl_connect('button_press_event', lambda event: on_press(event, self))
        self.canvas.mpl_connect('button_release_event', lambda event: on_release(event, self))
        self.precompile_numba_functions()

    def precompile_numba_functions(self):
        """Call numba-optimized functions with dummy data to precompile them."""
        calculate_implied_volatility_lr(0.1, 100.0, 100.0, 0.01, 0.5, option_type='calls')
        calculate_implied_volatility_baw(0.1, 100.0, 100.0, 0.01, 0.5, option_type='calls')

    def create_dropdown_and_button(self):
        dropdown_frame = tk.Frame(self.root)
        dropdown_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        left_frame = tk.Frame(dropdown_frame)
        left_frame.pack(side=tk.LEFT, anchor=tk.W)
        self.coord_label = tk.Label(left_frame, text="X: N/A    Y: N/A", fg='black', bg=dropdown_frame.cget('background'))
        self.coord_label.pack(side=tk.LEFT, padx=5)
        self.metrics_text = tk.Label(left_frame, text="χ²: N/A    avE5: N/A bps", fg='black', bg=dropdown_frame.cget('background'))
        self.metrics_text.pack(side=tk.LEFT, padx=10)
        right_frame = tk.Frame(dropdown_frame)
        right_frame.pack(side=tk.RIGHT, anchor=tk.E)
        metrics_frame = tk.Frame(right_frame)
        metrics_frame.pack(side=tk.LEFT, padx=5)
        selection_and_metrics_frame = tk.Frame(right_frame)
        selection_and_metrics_frame.pack(side=tk.LEFT)
        self.fit_var = tk.BooleanVar(value=True)
        self.fit_checkbox = tk.Checkbutton(selection_and_metrics_frame, text="Fit:", variable=self.fit_var)
        self.fit_checkbox.pack(side=tk.LEFT, padx=5)
        self.method_menu = ttk.Combobox(selection_and_metrics_frame, textvariable=self.selected_method, 
                                        values=["RBF", "RFV", "SVI", "SLV", "SABR"], state="readonly", style="TCombobox")
        self.method_menu.pack(side=tk.LEFT, padx=5)
        tk.Label(selection_and_metrics_frame, text="Obj. Function:").pack(side=tk.LEFT, padx=5)
        self.objective_menu = ttk.Combobox(selection_and_metrics_frame, textvariable=self.selected_objective, 
                                        values=["WLS", "WRE", "LS", "RE"], state="readonly", style="TCombobox")
        self.objective_menu.pack(side=tk.LEFT, padx=5)
        tk.Label(selection_and_metrics_frame, text="Epsilon:").pack(side=tk.LEFT, padx=5)
        self.epsilon_var = tk.StringVar(value="0.5")
        self.epsilon_entry = tk.Entry(selection_and_metrics_frame, textvariable=self.epsilon_var, width=10)
        self.epsilon_entry.pack(side=tk.LEFT, padx=5)
        tk.Label(selection_and_metrics_frame, text="IV Model:").pack(side=tk.LEFT, padx=5)
        self.pricing_model_menu = ttk.Combobox(selection_and_metrics_frame, textvariable=self.selected_pricing_model,
                                            values=["Leisen-Reimer", "Barone-Adesi Whaley"], state="readonly", style="TCombobox")
        self.pricing_model_menu.pack(side=tk.LEFT, padx=5)
        tk.Label(selection_and_metrics_frame, text="Mispricing:").pack(side=tk.LEFT, padx=5)
        self.mispricing_var = tk.StringVar(value="0.0")
        self.mispricing_entry = tk.Entry(selection_and_metrics_frame, textvariable=self.mispricing_var, width=10)
        self.mispricing_entry.pack(side=tk.LEFT, padx=5)
        tk.Label(selection_and_metrics_frame, text="Max Spread:").pack(side=tk.LEFT, padx=5)
        self.spread_filter_var = tk.StringVar(value="0.0")
        self.spread_filter_entry = tk.Entry(selection_and_metrics_frame, textvariable=self.spread_filter_var, width=10)
        self.spread_filter_entry.pack(side=tk.LEFT, padx=5)
        tk.Label(selection_and_metrics_frame, text="Strike Filter:").pack(side=tk.LEFT, padx=5)
        self.strike_filter_var = tk.StringVar(value="0.0")
        self.strike_filter_entry = tk.Entry(selection_and_metrics_frame, textvariable=self.strike_filter_var, width=10)
        self.strike_filter_entry.pack(side=tk.LEFT, padx=5)
        self.liquidity_filter_var = tk.BooleanVar(value=True)
        self.liquidity_filter_checkbox = tk.Checkbutton(selection_and_metrics_frame, text="Liquidity Filter", variable=self.liquidity_filter_var)
        self.liquidity_filter_checkbox.pack(side=tk.LEFT, padx=5)
        tk.Label(selection_and_metrics_frame, text="Exp. Date:").pack(side=tk.LEFT, padx=5)
        self.exp_date_var = tk.StringVar(value=self.expiration_dates_list[0])
        self.exp_date_menu = ttk.Combobox(selection_and_metrics_frame, textvariable=self.exp_date_var, 
                                        values=self.expiration_dates_list, state="readonly", style="TCombobox")
        self.exp_date_menu.pack(side=tk.LEFT, padx=5)
        tk.Label(selection_and_metrics_frame, text="Type:").pack(side=tk.LEFT, padx=5)
        self.type_var = tk.StringVar(value="calls")
        self.type_menu = ttk.Combobox(selection_and_metrics_frame, textvariable=self.type_var, 
                                    values=["calls", "puts"], state="readonly", style="TCombobox")
        self.type_menu.pack(side=tk.LEFT, padx=5)
        self.bid_var = tk.BooleanVar(value=True)
        self.bid_checkbox = tk.Checkbutton(selection_and_metrics_frame, text="Bid", variable=self.bid_var)
        self.bid_checkbox.pack(side=tk.LEFT, padx=5)
        self.ask_var = tk.BooleanVar(value=True)
        self.ask_checkbox = tk.Checkbutton(selection_and_metrics_frame, text="Ask", variable=self.ask_var)
        self.ask_checkbox.pack(side=tk.LEFT, padx=5)
        tk.Button(selection_and_metrics_frame, text="Enter", command=self.update_plot).pack(side=tk.LEFT, padx=5)

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
        data_dict = self.data_gen.data
        sorted_data = dict(sorted(data_dict.items()))

        try:
            strike_filter_value = float(self.strike_filter_var.get())
            if strike_filter_value < 0.0:
                raise ValueError("Strike Filter must be 0.0 or above.")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number for Strike Filter (0.0 or above).")
            return
        
        try:
            mispricing_value = float(self.mispricing_var.get())
            if mispricing_value < 0.0:
                raise ValueError("Mispricing must be 0.0 or above.")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number for Mispricing (0.0 or above).")
            return
        
        try:
            max_spread = float(self.spread_filter_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number for Max Bid-Ask Spread.")
            return

        if self.liquidity_filter_var.get():
            sorted_data = {strike: prices for strike, prices in sorted_data.items() if prices['bid'] != 0.0}

        try:
            epsilon_value = float(self.epsilon_var.get())
            if epsilon_value < 0.0:
                raise ValueError("Epsilon must be 0.0 or above.")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number for Epsilon (0.0 or above).")
            return    

        S = 176.94
        T = 0.030540532315417302
        r = self.risk_free_rate

        # Process original sorted_data_lr (calculate IVs for bid, ask, and mid prices using the selected pricing model)
        for strike, prices in sorted_data.items():
            if self.selected_pricing_model.get() == "Leisen-Reimer":
                pricing_model_function = calculate_implied_volatility_lr
            else:
                pricing_model_function = calculate_implied_volatility_baw
            sorted_data[strike] = {
                price_type: pricing_model_function(price, S, strike, r, T, option_type=self.type_var.get())
                for price_type, price in prices.items()
            }

        strike_prices = np.array(list(sorted_data.keys()))
        if strike_filter_value > 0.0:
            x = filter_strikes(strike_prices, np.mean(strike_prices), num_stdev=strike_filter_value)
        else:
            x = strike_prices
            
        sorted_data = {strike: prices for strike, prices in sorted_data.items() if strike in x}
        y_bid = np.array([prices['bid'] for prices in sorted_data.values()])
        y_ask = np.array([prices['ask'] for prices in sorted_data.values()])
        y_mid = np.array([prices['mid'] for prices in sorted_data.values()])
        
        if (max_spread > 0.0):
            mask = (y_ask - y_bid) <= max_spread
            x = x[mask]
            y_mid = y_mid[mask]
            y_bid = y_bid[mask]
            y_ask = y_ask[mask]
        if len(x) == 0:
            messagebox.showwarning("No Data", "All data points were filtered out. Adjust the spread filter.")
            return
        if hasattr(self, 'midpoints') and self.midpoints:
            self.midpoints.remove()
            self.midpoints = None
        if hasattr(self, 'outliers') and self.outliers:
            self.outliers.remove()
            self.outliers = None
        if hasattr(self, 'bids') and self.bids:
            self.bids.remove()
            self.bids = None
        if hasattr(self, 'asks') and self.asks:
            self.asks.remove()
            self.asks = None
        if hasattr(self, 'bid_lines'):
            for line in self.bid_lines:
                line.remove()
            self.bid_lines = []
        if hasattr(self, 'ask_lines'):
            for line in self.ask_lines:
                line.remove()
            self.ask_lines = []

        # Normalize X values here
        scaler = MinMaxScaler()
        x_normalized = scaler.fit_transform(x.reshape(-1, 1)).flatten()
        x_normalized = x_normalized + 0.5

        model = {
            "SVI": svi_model,
            "SLV": slv_model,
            "RFV": rfv_model,
            "SABR": sabr_model,
            "RBF": rbf_model,
        }.get(self.selected_method.get())

        if self.selected_method.get() == "RBF":
            smoothing = 0.000000000001
            interpolator = model(np.log(x_normalized), y_mid, epsilon=epsilon_value, smoothing=smoothing)
            fine_x_normalized = np.linspace(np.min(x_normalized), np.max(x_normalized), 400)
            interpolated_y = interpolator(np.log(fine_x_normalized).reshape(-1, 1))
            chi_squared = np.sum((y_mid - interpolator(np.log(x_normalized).reshape(-1, 1))) ** 2)
            avE5 = np.mean(np.abs(y_mid - interpolator(np.log(x_normalized).reshape(-1, 1)))) * 10000
        else:
            params = fit_model(x_normalized, y_mid, y_bid, y_ask, model, method=self.selected_objective.get())
            fine_x_normalized = np.linspace(np.min(x_normalized), np.max(x_normalized), 400)
            interpolated_y = model(np.log(fine_x_normalized), params)
            chi_squared, avE5 = compute_metrics(x_normalized, y_mid, model, params)

        fine_x = np.linspace(np.min(x), np.max(x), 400)
        outliers_indices = []
        if mispricing_value > 0.0:
            for i, x_value in enumerate(x):
                closest_index = np.argmin(np.abs(fine_x - x_value))
                interpolated_y_value = interpolated_y[closest_index]
                y_mid_value = data_dict[x_value]['mid']
                
                if self.selected_pricing_model.get() == "Leisen-Reimer":
                    option_price = leisen_reimer_tree(S, x_value, T, r, interpolated_y_value, option_type=self.type_var.get())
                else:
                    option_price = barone_adesi_whaley_american_option_price(S, x_value, T, r, interpolated_y_value, option_type=self.type_var.get())
                
                diff = abs(y_mid_value - option_price)
                
                if diff > mispricing_value:
                    outliers_indices.append(i)

        self.metrics_text.config(text=f"χ²: {chi_squared:.4f}    avE5: {avE5:.2f} bps")
        self.midpoints = self.ax.scatter(x, y_mid, color='red', s=20, label="Midpoint")

        if outliers_indices:
            self.outliers = self.ax.scatter(x[outliers_indices], y_mid[outliers_indices], color='yellow', s=20, label="Outliers")

        if self.bid_var.get():
            self.bids = self.ax.scatter(x, y_bid, color='red', s=10, label="Bid")
            self.bid_lines = [self.ax.plot([x[i], x[i]], [y_bid[i], y_mid[i]], color='red', linewidth=0.5)[0] for i in range(len(x))]
        else:
            self.bids = None
            self.bid_lines = []

        if self.ask_var.get():
            self.asks = self.ax.scatter(x, y_ask, color='red', s=10, label="Ask")
            self.ask_lines = [self.ax.plot([x[i], x[i]], [y_mid[i], y_ask[i]], color='red', linewidth=0.5)[0] for i in range(len(x))]
        else:
            self.asks = None
            self.ask_lines = []

        if hasattr(self, 'fit_line'):
            self.fit_line.set_data(fine_x, interpolated_y)
        else:
            self.fit_line, = self.ax.plot(fine_x, interpolated_y, color='green', label="Fit", linewidth=1.5)

        if self.fit_var.get():
            self.fit_line.set_visible(True)
        else:
            self.fit_line.set_visible(False)

        self.canvas.draw()

    def update_data_and_plot(self):
        self.data_gen.update_data()
        self.update_plot()
        self.root.after(10000, self.update_data_and_plot)  # 10 seconds interval

async def stream_live_prices(session, subs_list):
    async with DXLinkStreamer(session) as streamer:
        await streamer.subscribe(EventType.QUOTE, subs_list)
        start_time = time.time()
        while True:
            quote = await streamer.get_event(EventType.QUOTE)
            process_quote(quote)
            # Check if 1 second has passed
            if time.time() - start_time >= 1:
                start_time = time.time()

async def stream_raw_quotes(session, ticker_list):
    async with DXLinkStreamer(session) as streamer:
        await streamer.subscribe(EventType.QUOTE, ticker_list)
        while True:
            quote = await streamer.get_event(EventType.QUOTE)
            update_mid_price(quote)

def update_mid_price(quote):
    """
    Update the global underlying_price based on the new quote.
    
    Args:
        quote: The incoming quote object.
    """
    bid_price = quote.bidPrice
    ask_price = quote.askPrice
    underlying_price = float(math.floor((bid_price + ask_price) / 2 * 100) / 100)

def process_quote(quote):
    """
    Process incoming quote and update the data structure.

    Args:
        quote: Incoming quote object.
    """
    event_symbol = quote.eventSymbol
    bid_price = quote.bidPrice
    ask_price = quote.askPrice
    #strike_price = streamer_to_strike_map.get(event_symbol)

    #if strike_price is not None:
    #    mid_price = float(math.floor((bid_price + ask_price) / 2 * 100) / 100)
    #    quote_data[float(strike_price)] = {
    #        "bid": float(bid_price),
    #        "ask": float(ask_price),
    #        "mid": float(mid_price)
    #    }

def open_plot_manager(ticker, session, expiration_to_strikes_map, streamer_to_strike_map, expiration_dates_list, risk_free_rate):
    root = tk.Tk()
    PlotManager(root, ticker, session, expiration_to_strikes_map, streamer_to_strike_map, expiration_dates_list, risk_free_rate)
    root.mainloop()
