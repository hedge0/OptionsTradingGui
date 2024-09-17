import tkinter as tk
from tkinter import messagebox
from tastytrade import Session
from tastytrade.utils import TastytradeError
from tastytrade.instruments import NestedOptionChain
from fredapi import Fred
from credential_manager import load_cached_credentials, save_cached_credentials
from plot_manager import open_plot_manager

config = load_cached_credentials()
session = None
chain = None
expiration_to_strikes_map = {}
streamer_to_strike_map = {}
expiration_dates_list = []
risk_free_rate = 0.0

def show_initial_window():
    """
    Display the initial window with a select menu, FRED API key entry, and an Enter button.
    Hitting Enter validates the FRED API key and proceeds to the login elements.
    """
    initial_window = tk.Tk()
    initial_window.title("Select Platform")
    initial_window.geometry("300x250")  # Adjusted height to accommodate extra widgets

    # Platform Selection
    tk.Label(initial_window, text="Select Platform:").pack(pady=5)
    platform_var = tk.StringVar(value="TastyTrade")
    platform_menu = tk.OptionMenu(initial_window, platform_var, "TastyTrade")
    platform_menu.pack(pady=5)

    # FRED API Key Entry
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

        # Validate FRED API Key
        try:
            fred = Fred(api_key=fred_api_key)
            sofr_data = fred.get_series('SOFR')
            risk_free_rate = sofr_data.iloc[-1]

            if fred_remember_var.get():
                # Update cached credentials with FRED API key
                save_cached_credentials(config.get('TASTYTRADE_USERNAME'), config.get('TASTYTRADE_PASSWORD'), fred_api_key)

            if selected_platform == "TastyTrade":
                # Clear the window and show login elements
                show_login(initial_window)
        except Exception as e:
            messagebox.showerror("FRED API Error", f"Invalid FRED API Key: {str(e)}")
            return

    tk.Button(initial_window, text="Enter", command=proceed_to_login).pack(pady=20)

    initial_window.mainloop()

def show_login(window):
    """
    Display the login elements on the given window for the user to enter their credentials.

    This function adds widgets to the provided window where the user can input their Tastytrade username and password.
    It handles login validation and manages the caching of credentials.
    Upon successful login, it proceeds to validate the ticker and then shows options for expiration dates and option types.
    """
    # Clear the window
    for widget in window.winfo_children():
        widget.destroy()

    window.title("Login")
    window.geometry("300x250")

    tk.Label(window, text="Username:").pack(pady=5)
    username_entry = tk.Entry(window)
    username_entry.pack(pady=5)

    if config.get('TASTYTRADE_USERNAME'):
        username_entry.insert(0, config['TASTYTRADE_USERNAME'])

    tk.Label(window, text="Password:").pack(pady=5)
    password_entry = tk.Entry(window, show="*")
    password_entry.pack(pady=5)

    if config.get('TASTYTRADE_PASSWORD'):
        password_entry.insert(0, config['TASTYTRADE_PASSWORD'])

    remember_var = tk.BooleanVar(value=False)
    remember_checkbox = tk.Checkbutton(window, text="Remember Me", variable=remember_var)
    remember_checkbox.pack(pady=5)

    tk.Button(
        window,
        text="Login",
        command=lambda: check_credentials(window, username_entry, password_entry, remember_var)
    ).pack(pady=20)

def check_credentials(window, username_entry, password_entry, remember_var):
    """
    Validate the user's credentials and initiate a session with Tastytrade.

    This function checks the user's inputted username and password by attempting to create a session
    with Tastytrade. If successful, it will cache the credentials if the 'Remember Me' checkbox is selected.
    Upon successful login, it proceeds to the next step.
    """
    global session
    username = username_entry.get()
    password = password_entry.get()

    try:
        session = Session(login=username, password=password, remember_me=True)
        messagebox.showinfo("Login Success", "Login successful!")

        if remember_var.get():
            save_cached_credentials(username, password, config.get('FRED_API_KEY'))

        show_ticker_entry(window)
    except TastytradeError as e:
        if "invalid_credentials" in str(e):
            messagebox.showerror("Login Failed", "Incorrect Username or Password")
        else:
            messagebox.showerror("Login Failed", f"An error occurred: {str(e)}")

def show_ticker_entry(window):
    """
    Display the ticker entry field and the 'Search' button.
    After successful validation, display expiration date and option type selection below.
    """
    # Clear the window
    for widget in window.winfo_children():
        widget.destroy()

    window.title("Ticker Entry")
    window.geometry("300x400")  # Adjusted height to accommodate additional widgets

    tk.Label(window, text="Ticker:").pack(pady=5)
    ticker_entry = tk.Entry(window)
    ticker_entry.pack(pady=5)

    # Create the dynamic widgets frame but do not pack it yet
    dynamic_widgets_frame = tk.Frame(window)

    # Pack the 'Search' button above the dynamic widgets frame
    tk.Button(
        window,
        text="Search",
        command=lambda: validate_ticker(window, ticker_entry, dynamic_widgets_frame)
    ).pack(pady=20)

    # Now pack the dynamic widgets frame after the 'Search' button with extra padding
    dynamic_widgets_frame.pack(pady=(40, 0))  # Increased pady for more space

def validate_ticker(window, ticker_entry, dynamic_widgets_frame):
    """
    Validate the ticker symbol and retrieve option chains.
    If successful, display expiration date and option type selection below.
    """
    global chain, expiration_to_strikes_map, streamer_to_strike_map, expiration_dates_list
    ticker = ticker_entry.get()

    # Clear previous dynamic widgets if any
    for widget in dynamic_widgets_frame.winfo_children():
        widget.destroy()

    try:
        chain = NestedOptionChain.get_chain(session, ticker)

        if chain is not None:
            expiration_to_strikes_map = {}
            streamer_to_strike_map = {}
            expiration_dates_list = []

            for expiration in chain.expirations:
                calls_list = []
                puts_list = []

                for strike in expiration.strikes:
                    calls_list.append(strike.call_streamer_symbol)
                    puts_list.append(strike.put_streamer_symbol)

                    streamer_to_strike_map[strike.call_streamer_symbol] = strike.strike_price
                    streamer_to_strike_map[strike.put_streamer_symbol] = strike.strike_price

                expiration_to_strikes_map[expiration.expiration_date] = {
                    "calls": calls_list,
                    "puts": puts_list
                }
                expiration_dates_list.append(expiration.expiration_date)

            # Now, display expiration date and option type selection below the ticker entry
            show_expiration_and_option_type_selection(dynamic_widgets_frame, ticker, window)
    except TastytradeError as e:
        if "record_not_found" in str(e):
            messagebox.showerror("Validation Failed", f"Invalid ticker symbol: {ticker}. Please check and try again.")
            return
        else:
            messagebox.showerror("Validation Failed", f"An error occurred: {str(e)}")
            return

def show_expiration_and_option_type_selection(frame, ticker, window):
    """
    Display the expiration date and option type selection below the ticker entry.
    """
    # Do not clear the window; add widgets to the provided frame

    tk.Label(frame, text="Select Expiration Date:").pack(pady=5)
    expiration_var = tk.StringVar(value=expiration_dates_list[0])
    expiration_menu = tk.OptionMenu(frame, expiration_var, *expiration_dates_list)
    expiration_menu.pack(pady=5)

    tk.Label(frame, text="Select Option Type:").pack(pady=5)
    option_type_var = tk.StringVar(value="calls")
    option_type_menu = tk.OptionMenu(frame, option_type_var, "calls", "puts")
    option_type_menu.pack(pady=5)

    # Add an 'Enter' button
    tk.Button(
        frame,
        text="Enter",
        command=lambda: proceed_to_plot(
            window, ticker, expiration_var.get(), option_type_var.get()
        )
    ).pack(pady=20)

def proceed_to_plot(window, ticker, selected_date, option_type):
    """
    Proceed to plot the selected options data.

    This function is called when the user selects an expiration date and option type,
    and clicks 'Enter'. It destroys the window and opens the plot manager with the
    selected data.
    """
    window.destroy()
    open_plot_manager(
        ticker,
        session,
        expiration_to_strikes_map,
        streamer_to_strike_map,
        selected_date,
        option_type,
        risk_free_rate
    )

if __name__ == "__main__":
    show_initial_window()
