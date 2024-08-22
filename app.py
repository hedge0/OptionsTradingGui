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

def show_login():
    login_window = tk.Tk()
    login_window.title("Login")
    login_window.geometry("300x250")

    tk.Label(login_window, text="Username:").pack(pady=5)
    username_entry = tk.Entry(login_window)
    username_entry.pack(pady=5)

    if config.get('TASTYTRADE_USERNAME'):
        username_entry.insert(0, config['TASTYTRADE_USERNAME'])

    tk.Label(login_window, text="Password:").pack(pady=5)
    password_entry = tk.Entry(login_window, show="*")
    password_entry.pack(pady=5)

    if config.get('TASTYTRADE_PASSWORD'):
        password_entry.insert(0, config['TASTYTRADE_PASSWORD'])

    remember_var = tk.BooleanVar(value=False)
    remember_checkbox = tk.Checkbutton(login_window, text="Remember Me", variable=remember_var)
    remember_checkbox.pack(pady=5)

    def check_credentials():
        global session, chain
        username = username_entry.get()
        password = password_entry.get()

        try:
            session = Session(login=username, password=password, remember_me=True)
            messagebox.showinfo("Login Success", "Login successful!")

            if remember_var.get():
                save_cached_credentials(username, password)

            for widget in login_window.winfo_children():
                widget.pack_forget()

            tk.Label(login_window, text="Ticker:").pack(pady=5)
            ticker_entry = tk.Entry(login_window)
            ticker_entry.pack(pady=5)
            
            tk.Label(login_window, text="FRED API Key:").pack(pady=5)
            fred_api_key_entry = tk.Entry(login_window)
            fred_api_key_entry.pack(pady=5)

            if config.get('FRED_API_KEY'):
                fred_api_key_entry.insert(0, config['FRED_API_KEY'])

            fred_remember_var = tk.BooleanVar(value=False)
            fred_remember_checkbox = tk.Checkbutton(login_window, text="Remember Me", variable=fred_remember_var)
            fred_remember_checkbox.pack(pady=5)

            def validate_and_show_options():
                global chain, expiration_to_strikes_map, streamer_to_strike_map, expiration_dates_list, risk_free_rate
                ticker = ticker_entry.get()
                fred_api_key = fred_api_key_entry.get()

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

                except TastytradeError as e:
                    if "record_not_found" in str(e):
                        messagebox.showerror("Validation Failed", f"Invalid ticker symbol: {ticker}. Please check and try again.")
                        return
                    else:
                        messagebox.showerror("Validation Failed", f"An error occurred: {str(e)}")
                        return
                    
                try:
                    if fred_remember_var.get():
                        save_cached_credentials(username, password, fred_api_key)

                    fred = Fred(api_key=fred_api_key)
                    sofr_data = fred.get_series('SOFR')
                    risk_free_rate = sofr_data.iloc[-1]
                except Exception as e:
                    messagebox.showerror("FRED API Error", f"Invalid FRED API Key: {str(e)}")
                    return

                for widget in login_window.winfo_children():
                    widget.pack_forget()

                tk.Label(login_window, text="Select Expiration Date:").pack(pady=5)
                expiration_var = tk.StringVar(value=expiration_dates_list[0])
                expiration_menu = tk.OptionMenu(login_window, expiration_var, *expiration_dates_list)
                expiration_menu.pack(pady=5)

                tk.Label(login_window, text="Select Option Type:").pack(pady=5)
                option_type_var = tk.StringVar(value="calls")
                option_type_menu = tk.OptionMenu(login_window, option_type_var, "calls", "puts")
                option_type_menu.pack(pady=5)

                def proceed_to_plot():
                    selected_date = expiration_var.get()
                    option_type = option_type_var.get()
                    login_window.destroy()
                    open_plot_manager(ticker, session, expiration_to_strikes_map, streamer_to_strike_map, selected_date, option_type, risk_free_rate)

                tk.Button(login_window, text="Enter", command=proceed_to_plot).pack(pady=20)

            tk.Button(login_window, text="Enter", command=validate_and_show_options).pack(pady=20)

        except TastytradeError as e:
            if "invalid_credentials" in str(e):
                messagebox.showerror("Login Failed", "Incorrect Username or Password")
            else:
                messagebox.showerror("Login Failed", f"An error occurred: {str(e)}")

    tk.Button(login_window, text="Login", command=check_credentials).pack(pady=20)

    login_window.mainloop()

if __name__ == "__main__":
    show_login()
