import os
from dotenv import load_dotenv
from tastytrade import Session, Account
from tastytrade.utils import TastytradeError
from tastytrade.instruments import NestedOptionChain

# Load environment variables from .env file
load_dotenv()

# Constants and Global Variables
config = {}

# Global variables for Tastytrade
session = None
account = None
chain = None  # Declare chain as a global variable

def load_config():
    """
    Load configuration from environment variables and validate them.
    
    Raises:
        ValueError: If any required environment variable is not set.
    """
    global config
    config = {
        "TASTYTRADE_USERNAME": os.getenv('TASTYTRADE_USERNAME'),
        "TASTYTRADE_PASSWORD": os.getenv('TASTYTRADE_PASSWORD'),
        "TASTYTRADE_ACCOUNT_NUMBER": os.getenv('TASTYTRADE_ACCOUNT_NUMBER'),
    }

    for key, value in config.items():
        if value is None:
            raise ValueError(f"{key} environment variable not set")
        
def main():
    """
    Main function to initialize the bot.
    """
    global session, account, chain

    load_config()

    try:
        session = Session(login=config["TASTYTRADE_USERNAME"], password=config["TASTYTRADE_PASSWORD"], remember_me=True)
        print("Login successful.")
    except TastytradeError as e:
        if "invalid_credentials" in str(e):
            print("Invalid login credentials. Please check your username and password.")
            return
        else:
            raise  # Re-raise if the error is not about invalid credentials

    try:
        account = Account.get_account(session, config["TASTYTRADE_ACCOUNT_NUMBER"])
    except TastytradeError as e:
        if "record_not_found" in str(e):
            print(f"Invalid account number: {config['TASTYTRADE_ACCOUNT_NUMBER']}. Please check the account number and try again.")
            return
        else:
            raise  # Re-raise if the error is not specifically about the account number

    try:
        ticker = 'AMZN'
        chain = NestedOptionChain.get_chain(session, ticker)
        print(f"Option chain for {ticker} retrieved successfully.")
    except TastytradeError as e:
        if "record_not_found" in str(e):
            print(f"Invalid ticker symbol: {ticker}. Please try again.")
            chain = None
        else:
            raise  # Re-raise if the error is not specifically about the ticker symbol

    if chain is not None:
        # Create the nested expiration-to-strikes map and array of dates
        expiration_to_strikes_map = {}
        expiration_dates_list = []

        for expiration in chain.expirations:
            calls_map = {}
            puts_map = {}

            for strike in expiration.strikes:
                # Populate the calls map
                calls_map[strike.strike_price] = (strike.strike_price, strike.call_streamer_symbol)
                
                # Populate the puts map
                puts_map[strike.strike_price] = (strike.strike_price, strike.put_streamer_symbol)

            # Map each expiration date to the calls and puts map
            expiration_to_strikes_map[expiration.expiration_date] = {
                "calls": calls_map,
                "puts": puts_map
            }
            # Add the expiration date to the list
            expiration_dates_list.append(expiration.expiration_date)

        # Print the array of expiration dates

        # Print the resulting map for visualization
        for expiration_date, option_types in expiration_to_strikes_map.items():
            print(f"Expiration Date: {expiration_date}")
            print("Calls:")
            for strike, call_data in option_types['calls'].items():
                print(f"  Strike: {strike}, Call Streamer Symbol: {call_data[1]}")
            print("Puts:")
            for strike, put_data in option_types['puts'].items():
                print(f"  Strike: {strike}, Put Streamer Symbol: {put_data[1]}")
            print()

        print("Expiration Dates List:", expiration_dates_list)

if __name__ == '__main__':
    main()
