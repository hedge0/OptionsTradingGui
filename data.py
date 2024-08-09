import os
from dotenv import load_dotenv
from tastytrade import Session, Account
from tastytrade.utils import TastytradeError

# Load environment variables from .env file
load_dotenv()

# Constants and Global Variables
config = {}

# Global variables for Tastytrade
session = None
account = None

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
    global session, account

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

    account = Account.get_account(session, config["TASTYTRADE_ACCOUNT_NUMBER"])

    print(account)

if __name__ == '__main__':
    main()