import tkinter as tk
from tkinter import messagebox
import httpx
from tastytrade import Session
from tastytrade.utils import TastytradeError
from tastytrade.instruments import NestedOptionChain
from fredapi import Fred
from schwab.auth import easy_client

import nest_asyncio
import threading
import asyncio

nest_asyncio.apply()

from credential_manager import load_cached_credentials, save_cached_credentials
from plot_manager_tasty import open_plot_manager_tasty

config = load_cached_credentials()
risk_free_rate = 0.0

def start_event_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

loop = asyncio.new_event_loop()
threading.Thread(target=start_event_loop, args=(loop,), daemon=True).start()

def show_initial_window():
    """
    Display the initial window with a select menu, FRED API key entry, and an Enter button.
    Hitting Enter validates the FRED API key and proceeds to the login elements.
    """
    initial_window = tk.Tk()
    initial_window.title("App Manager")
    initial_window.geometry("300x250")

    tk.Label(initial_window, text="Select Platform:").pack(pady=5)
    platform_var = tk.StringVar(value="TastyTrade")
    platform_menu = tk.OptionMenu(initial_window, platform_var, "TastyTrade", "Schwab")
    platform_menu.pack(pady=5)

    tk.Label(initial_window, text="FRED API Key:").pack(pady=5)
    fred_api_key_entry = tk.Entry(initial_window)
    fred_api_key_entry.pack(pady=5)

    if config.get('FRED_API_KEY'):
        fred_api_key_entry.insert(0, config['FRED_API_KEY'])

    fred_remember_var = tk.BooleanVar(value=False)
    fred_remember_checkbox = tk.Checkbutton(initial_window, text="Remember Me", variable=fred_remember_var)
    fred_remember_checkbox.pack(pady=5)

    def proceed_to_login():
        global risk_free_rate
        fred_api_key = fred_api_key_entry.get()
        selected_platform = platform_var.get()

        try:
            fred = Fred(api_key=fred_api_key)
            sofr_data = fred.get_series('SOFR')
            risk_free_rate = sofr_data.iloc[-1]

            if fred_remember_var.get():
                save_cached_credentials(
                    config.get('TASTYTRADE_USERNAME'), 
                    config.get('TASTYTRADE_PASSWORD'), 
                    fred_api_key, 
                    config.get('SCHWAB_API_KEY'), 
                    config.get('SCHWAB_SECRET'), 
                    config.get('SCHWAB_CALLBACK_URL')
                )

            if selected_platform == "TastyTrade":
                tastytrade_instance = Tastytrade(window=initial_window, risk_free_rate=risk_free_rate)
                tastytrade_instance.show_login()
            elif selected_platform == "Schwab":
                schwab_instance = Schwab(window=initial_window, risk_free_rate=risk_free_rate)
                schwab_instance.show_login()
            else:
                messagebox.showerror("Platform Error", f"The selected platform '{selected_platform}' is not supported.")
        except Exception as e:
            messagebox.showerror("FRED API Error", f"Invalid FRED API Key: {str(e)}")
            return

    tk.Button(initial_window, text="Enter", command=proceed_to_login).pack(pady=20)

    initial_window.mainloop()













class Tastytrade:
    def __init__(self, window, risk_free_rate):
        self.window = window
        self.risk_free_rate = risk_free_rate
        self.session = None
        self.chain = None
        self.expiration_to_strikes_map = {}
        self.streamer_to_strike_map = {}
        self.expiration_dates_list = []

    def show_login(self):
        """
        Display the login elements for Tastytrade on the given window for the user to enter their credentials.
        """
        for widget in self.window.winfo_children():
            widget.destroy()

        tk.Label(self.window, text="Username:").pack(pady=5)
        username_entry = tk.Entry(self.window)
        username_entry.pack(pady=5)

        if config.get('TASTYTRADE_USERNAME'):
            username_entry.insert(0, config['TASTYTRADE_USERNAME'])

        tk.Label(self.window, text="Password:").pack(pady=5)
        password_entry = tk.Entry(self.window, show="*")
        password_entry.pack(pady=5)

        if config.get('TASTYTRADE_PASSWORD'):
            password_entry.insert(0, config['TASTYTRADE_PASSWORD'])

        remember_var = tk.BooleanVar(value=False)
        remember_checkbox = tk.Checkbutton(self.window, text="Remember Me", variable=remember_var)
        remember_checkbox.pack(pady=5)

        tk.Button(
            self.window,
            text="Login",
            command=lambda: self.check_credentials(username_entry, password_entry, remember_var)
        ).pack(pady=20)

    def check_credentials(self, username_entry, password_entry, remember_var):
        """
        Validate the user's credentials and initiate a session with Tastytrade.
        """
        username = username_entry.get()
        password = password_entry.get()

        try:
            self.session = Session(login=username, password=password, remember_me=True)
            messagebox.showinfo("Login Success", "Login successful!")

            if remember_var.get():
                save_cached_credentials(
                    username=username,
                    password=password,
                    fred_api_key=config.get('FRED_API_KEY'),
                    schwab_api_key=config.get('SCHWAB_API_KEY'),
                    schwab_secret=config.get('SCHWAB_SECRET'),
                    schwab_callback_url=config.get('SCHWAB_CALLBACK_URL')
                )

            self.show_ticker_entry()
        except TastytradeError as e:
            if "invalid_credentials" in str(e):
                messagebox.showerror("Login Failed", "Incorrect Username or Password")
            else:
                messagebox.showerror("Login Failed", f"An error occurred: {str(e)}")

    def show_ticker_entry(self):
        """
        Display the ticker entry field and the 'Search' button.
        After successful validation, display expiration date and option type selection below.
        """
        for widget in self.window.winfo_children():
            widget.destroy()

        self.window.geometry("300x400")

        tk.Label(self.window, text="Ticker:").pack(pady=5)
        ticker_entry = tk.Entry(self.window)
        ticker_entry.pack(pady=5)

        dynamic_widgets_frame = tk.Frame(self.window)

        tk.Button(
            self.window,
            text="Search",
            command=lambda: self.validate_ticker(ticker_entry, dynamic_widgets_frame)
        ).pack(pady=20)

        dynamic_widgets_frame.pack(pady=(40, 0))

    def validate_ticker(self, ticker_entry, dynamic_widgets_frame):
        """
        Validate the ticker symbol and retrieve option chains.
        If successful, display expiration date and option type selection below.
        """
        ticker = ticker_entry.get()

        try:
            self.chain = NestedOptionChain.get_chain(self.session, ticker)

            for widget in dynamic_widgets_frame.winfo_children():
                widget.destroy()

            if self.chain is not None:
                self.expiration_to_strikes_map = {}
                self.streamer_to_strike_map = {}
                self.expiration_dates_list = []

                for expiration in self.chain.expirations:
                    calls_list = []
                    puts_list = []

                    for strike in expiration.strikes:
                        calls_list.append(strike.call_streamer_symbol)
                        puts_list.append(strike.put_streamer_symbol)

                        self.streamer_to_strike_map[strike.call_streamer_symbol] = strike.strike_price
                        self.streamer_to_strike_map[strike.put_streamer_symbol] = strike.strike_price

                    self.expiration_to_strikes_map[expiration.expiration_date] = {
                        "calls": calls_list,
                        "puts": puts_list
                    }
                    self.expiration_dates_list.append(expiration.expiration_date)

                self.show_expiration_and_option_type_selection(dynamic_widgets_frame, ticker)
        except TastytradeError as e:
            for widget in dynamic_widgets_frame.winfo_children():
                widget.destroy()
        
            if "record_not_found" in str(e):
                messagebox.showerror("Validation Failed", f"Invalid ticker symbol: {ticker}. Please check and try again.")
                return
            else:
                messagebox.showerror("Validation Failed", f"An error occurred: {str(e)}")
                return

    def show_expiration_and_option_type_selection(self, frame, ticker):
        """
        Display the expiration date and option type selection below the ticker entry.
        """
        tk.Label(frame, text="Select Expiration Date:").pack(pady=5)
        expiration_var = tk.StringVar(value=self.expiration_dates_list[0])
        expiration_menu = tk.OptionMenu(frame, expiration_var, *self.expiration_dates_list)
        expiration_menu.pack(pady=5)

        tk.Label(frame, text="Select Option Type:").pack(pady=5)
        option_type_var = tk.StringVar(value="calls")
        option_type_menu = tk.OptionMenu(frame, option_type_var, "calls", "puts")
        option_type_menu.pack(pady=5)

        tk.Button(
            frame,
            text="Enter",
            command=lambda: self.proceed_to_plot(
                ticker, expiration_var.get(), option_type_var.get()
            )
        ).pack(pady=20)

    def proceed_to_plot(self, ticker, selected_date, option_type):
        """
        Proceed to plot the selected options data.

        This function is called when the user selects an expiration date and option type,
        and clicks 'Enter'. It destroys the window and opens the plot manager with the
        selected data.
        """
        open_plot_manager_tasty(
            ticker,
            self.session,
            self.expiration_to_strikes_map,
            self.streamer_to_strike_map,
            selected_date,
            option_type,
            self.risk_free_rate
        )















class Schwab:
    def __init__(self, window, risk_free_rate):
        self.window = window
        self.risk_free_rate = risk_free_rate
        self.session = None
        self.expiration_dates_list = []

    def show_login(self):
        """
        Display the login elements for Schwab on the given window for the user to enter their credentials.
        """
        for widget in self.window.winfo_children():
            widget.destroy()

        self.window.geometry("300x300")

        tk.Label(self.window, text="API Key:").pack(pady=5)
        api_key_entry = tk.Entry(self.window)
        api_key_entry.pack(pady=5)

        if config.get('SCHWAB_API_KEY'):
            api_key_entry.insert(0, config['SCHWAB_API_KEY'])

        tk.Label(self.window, text="Secret:").pack(pady=5)
        secret_entry = tk.Entry(self.window, show="*")
        secret_entry.pack(pady=5)

        if config.get('SCHWAB_SECRET'):
            secret_entry.insert(0, config['SCHWAB_SECRET'])

        tk.Label(self.window, text="Callback URL:").pack(pady=5)
        callback_url_entry = tk.Entry(self.window)
        callback_url_entry.pack(pady=5)

        if config.get('SCHWAB_CALLBACK_URL'):
            callback_url_entry.insert(0, config['SCHWAB_CALLBACK_URL'])

        tk.Button(
            self.window,
            text="Login",
            command=lambda: self.check_credentials(api_key_entry, secret_entry, callback_url_entry)
        ).pack(pady=20)

    def check_credentials(self, api_key_entry, secret_entry, callback_url_entry):
        """
        Validate the user's credentials and initiate a session with Schwab.
        """
        api_key = api_key_entry.get()
        secret = secret_entry.get()
        callback_url = callback_url_entry.get()

        try:
            self.session = easy_client(
                token_path='token.json',
                api_key=api_key,
                app_secret=secret,
                callback_url=callback_url,
                asyncio=True)
            messagebox.showinfo("Login Success", "Login successful!")

            self.show_ticker_entry()
        except Exception as e:
            messagebox.showerror("Login Failed", f"An error occurred: {str(e)}")

    def show_ticker_entry(self):
        """
        Display the ticker entry field and the 'Search' button.
        """
        for widget in self.window.winfo_children():
            widget.destroy()

        self.window.geometry("300x400")

        tk.Label(self.window, text="Ticker:").pack(pady=5)
        ticker_entry = tk.Entry(self.window)
        ticker_entry.pack(pady=5)

        dynamic_widgets_frame = tk.Frame(self.window)

        tk.Button(
            self.window,
            text="Search",
            command=lambda: self.run_async_validation(ticker_entry, dynamic_widgets_frame)
        ).pack(pady=20)

        dynamic_widgets_frame.pack(pady=(40, 0))

    def run_async_validation(self, ticker_entry, dynamic_widgets_frame):
        """
        Run the asynchronous ticker validation using `asyncio.run_coroutine_threadsafe`.
        """
        asyncio.run_coroutine_threadsafe(self.validate_ticker(ticker_entry, dynamic_widgets_frame), loop)

    async def validate_ticker(self, ticker_entry, dynamic_widgets_frame):
        """
        Validate the ticker symbol and retrieve option chains.
        """
        ticker = ticker_entry.get()

        try:
            resp = await self.session.get_option_expiration_chain(ticker)

            for widget in dynamic_widgets_frame.winfo_children():
                widget.destroy()
            
            assert resp.status_code == httpx.codes.OK
            expirations = resp.json()

            if expirations is not None and expirations["expirationList"]:
                self.expiration_dates_list = []

                for expiration in expirations["expirationList"]:
                    self.expiration_dates_list.append(expiration["expirationDate"])

                self.show_expiration_and_option_type_selection(dynamic_widgets_frame, ticker)
            else:
                messagebox.showerror("Validation Failed", f"Invalid ticker symbol: {ticker}. Please check and try again.")
                return
        except Exception as e:
            for widget in dynamic_widgets_frame.winfo_children():
                widget.destroy()
        
            messagebox.showerror("Validation Failed", f"An error occurred: {str(e)}")
            return

    def show_expiration_and_option_type_selection(self, frame, ticker):
        """
        Display the expiration date and option type selection below the ticker entry.
        """
        tk.Label(frame, text="Select Expiration Date:").pack(pady=5)
        expiration_var = tk.StringVar(value=self.expiration_dates_list[0])
        expiration_menu = tk.OptionMenu(frame, expiration_var, *self.expiration_dates_list)
        expiration_menu.pack(pady=5)

        tk.Label(frame, text="Select Option Type:").pack(pady=5)
        option_type_var = tk.StringVar(value="calls")
        option_type_menu = tk.OptionMenu(frame, option_type_var, "calls", "puts")
        option_type_menu.pack(pady=5)





if __name__ == "__main__":
    show_initial_window()