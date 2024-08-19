import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from models import filter_strikes, svi_model, slv_model, rfv_model, sabr_model, fit_model, compute_metrics
from data_generator import DataGenerator
from sklearn.preprocessing import MinMaxScaler

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
        self.selected_method = tk.StringVar(value="RFV")
        self.selected_objective = tk.StringVar(value="WRE")
        self.ticker = ticker

        self.press_event = None  # To store the initial press event for dragging

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

        # Connect the scroll wheel for zooming
        self.canvas.mpl_connect('scroll_event', self.on_scroll)

        # Connect the motion event to update coordinates
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        # Connect mouse press and release events for dragging
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)

    def create_dropdown_and_button(self):
        dropdown_frame = tk.Frame(self.root)
        dropdown_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Frame for the coordinates and metrics labels (left-aligned)
        left_frame = tk.Frame(dropdown_frame)
        left_frame.pack(side=tk.LEFT, anchor=tk.W)

        # Label to display the coordinates with a larger gap between X and Y
        self.coord_label = tk.Label(left_frame, text="X: N/A    Y: N/A", fg='black', bg=dropdown_frame.cget('background'))
        self.coord_label.pack(side=tk.LEFT, padx=5)

        # Adjusted metrics label with a larger gap between coordinates and metrics
        self.metrics_text = tk.Label(left_frame, text="χ²: N/A    avE5: N/A bps", fg='black', bg=dropdown_frame.cget('background'))
        self.metrics_text.pack(side=tk.LEFT, padx=10)

        # Frame for the rest of the controls (right-aligned)
        right_frame = tk.Frame(dropdown_frame)
        right_frame.pack(side=tk.RIGHT, anchor=tk.E)

        # Metrics display frame (moved to the right)
        metrics_frame = tk.Frame(right_frame)
        metrics_frame.pack(side=tk.LEFT, padx=5)

        # Create a frame for the model selection and metrics
        selection_and_metrics_frame = tk.Frame(right_frame)
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

        # Add the Strike Filter input field
        tk.Label(selection_and_metrics_frame, text="Strike Filter:").pack(side=tk.LEFT, padx=5)
        self.strike_filter_var = tk.StringVar(value="0.0")
        self.strike_filter_entry = tk.Entry(selection_and_metrics_frame, textvariable=self.strike_filter_var, width=10)
        self.strike_filter_entry.pack(side=tk.LEFT, padx=5)

        # Add the Exp. Date dropdown menu
        tk.Label(selection_and_metrics_frame, text="Exp. Date:").pack(side=tk.LEFT, padx=5)
        self.exp_date_var = tk.StringVar(value=self.expiration_dates_list[0])  # Default to the first date
        self.exp_date_menu = ttk.Combobox(selection_and_metrics_frame, textvariable=self.exp_date_var, 
                                        values=self.expiration_dates_list, state="readonly", style="TCombobox")
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
        data_dict = self.data_gen.data
        sorted_data = dict(sorted(data_dict.items()))

        # Get the Strike Filter value and ensure it's a float 0.0 or above
        try:
            strike_filter_value = float(self.strike_filter_var.get())
            if strike_filter_value < 0.0:
                raise ValueError("Strike Filter must be 0.0 or above.")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number for Strike Filter (0.0 or above).")
            return

        # Extract x, y_bid, y_ask, and y_mid from the dictionary
        strike_prices = np.array(list(sorted_data.keys()))

        # Apply the strike filter only if strike_filter_value is greater than 0.0
        if strike_filter_value > 0.0:
            x = filter_strikes(strike_prices, np.mean(strike_prices), num_stdev=strike_filter_value)
        else:
            x = strike_prices
            
        sorted_data = {strike: prices for strike, prices in sorted_data.items() if strike in x}

        y_bid = np.array([prices['bid'] for prices in sorted_data.values()])
        y_ask = np.array([prices['ask'] for prices in sorted_data.values()])
        y_mid = np.array([prices['mid'] for prices in sorted_data.values()])

        # Apply the bid-ask spread filter if necessary
        try:
            max_spread = float(self.spread_filter_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number for Max Bid-Ask Spread.")
            return
        
        if max_spread > 0.0:
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

        # normalize X values here
        scaler = MinMaxScaler()
        x_normalized = scaler.fit_transform(x.reshape(-1, 1)).flatten()
        x_normalized = x_normalized + 1

        model = {
            "SVI": svi_model,
            "SLV": slv_model,
            "RFV": rfv_model,
            "SABR": sabr_model,
        }.get(self.selected_method.get())

        # Apply the selected objective function and fit the model
        params = fit_model(x_normalized, y_mid, y_bid, y_ask, model, method=self.selected_objective.get())
        fine_x_normalized = np.linspace(np.min(x_normalized), np.max(x_normalized), 200)
        fine_x = np.linspace(np.min(x), np.max(x), 200)
        interpolated_y = model(np.log(fine_x_normalized), params)
        chi_squared, avE5 = compute_metrics(x_normalized, y_mid, model, params)

        # Apply results to plot
        self.metrics_text.config(text=f"χ²: {chi_squared:.4f}    avE5: {avE5:.2f} bps")
        self.bids = self.ax.scatter(x, y_bid, color='red', s=10, label="Bid")
        self.asks = self.ax.scatter(x, y_ask, color='red', s=10, label="Ask")
        self.midpoints = self.ax.scatter(x, y_mid, color='red', s=20, label="Midpoint")
        self.lines = [self.ax.plot([x[i], x[i]], [y_bid[i], y_ask[i]], color='red', linewidth=0.5)[0] for i in range(len(x))]

        if hasattr(self, 'fit_line'):
            self.fit_line.set_data(fine_x, interpolated_y)
        else:
            self.fit_line, = self.ax.plot(fine_x, interpolated_y, color='green', label="Fit", linewidth=1.5)

        self.canvas.draw()


    def update_data_and_plot(self):
        self.data_gen.update_data()
        self.update_plot()
        self.root.after(1000000, self.update_data_and_plot)  # 10 seconds interval

    def on_mouse_move(self, event):
        if event.inaxes:
            # Ensure the event.xdata and event.ydata are within bounds
            if event.xdata is not None and event.ydata is not None:
                x_coord = f"{event.xdata:.2f}"
                y_coord = f"{event.ydata:.4f}"
                self.coord_label.config(text=f"X: {x_coord}    Y: {y_coord}")

            # If dragging, update the view limits
            if self.press_event is not None:
                dx = event.xdata - self.press_event.xdata
                dy = event.ydata - self.press_event.ydata
                cur_xlim = self.ax.get_xlim()
                cur_ylim = self.ax.get_ylim()
                self.ax.set_xlim(cur_xlim[0] - dx, cur_xlim[1] - dx)
                self.ax.set_ylim(cur_ylim[0] - dy, cur_ylim[1] - dy)
                self.canvas.draw()

    def on_scroll(self, event):
        """Zoom in/out with the scroll wheel."""
        # Check if the event's xdata and ydata are not None
        if event.xdata is None or event.ydata is None:
            return  # Exit the function if the event occurs outside the axes

        base_scale = 1.2

        # Get the current x and y limits
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        # Calculate the scaling factor
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            scale_factor = 1

        # Calculate the new limits
        new_xlim = [event.xdata - (event.xdata - cur_xlim[0]) * scale_factor,
                    event.xdata + (cur_xlim[1] - event.xdata) * scale_factor]
        new_ylim = [event.ydata - (event.ydata - cur_ylim[0]) * scale_factor,
                    event.ydata + (cur_ylim[1] - event.ydata) * scale_factor]

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.canvas.draw()

    def on_press(self, event):
        """Store the initial press event for dragging."""
        if event.inaxes:
            self.press_event = event

    def on_release(self, event):
        """Reset the press_event after releasing the mouse button."""
        self.press_event = None

def open_plot_manager(ticker, session, expiration_to_strikes_map, streamer_to_strike_map, expiration_dates_list, risk_free_rate):
    root = tk.Tk()
    PlotManager(root, ticker, session, expiration_to_strikes_map, streamer_to_strike_map, expiration_dates_list, risk_free_rate)
    root.mainloop()
