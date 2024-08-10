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
chain = None

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
    except TastytradeError as e:
        if "record_not_found" in str(e):
            print(f"Invalid ticker symbol: {ticker}. Please try again.")
            return
        else:
            raise  # Re-raise if the error is not specifically about the ticker symbol

    if chain is not None:
        print(chain.expirations[0].expiration_date)

if __name__ == '__main__':
    main()
