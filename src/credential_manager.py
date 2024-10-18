import os
import json

CACHE_FILE = 'credentials_cache.json'

def load_cached_credentials():
    """
    Load cached credentials from a file.

    Returns:
        dict: A dictionary with the cached credentials if the file exists, otherwise an empty dictionary.
    """
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as file:
            return json.load(file)
    return {}

def save_cached_credentials(username=None, password=None, fred_api_key=None, schwab_api_key=None, schwab_secret=None, schwab_callback_url=None):
    """
    Save the provided username, password, FRED API Key, and Schwab credentials to the cache file.

    Args:
        username (str): The username to cache.
        password (str): The password to cache.
        fred_api_key (str): The FRED API Key to cache.
        schwab_api_key (str): The Schwab API Key to cache.
        schwab_secret (str): The Schwab secret key to cache.
        schwab_callback_url (str): The Schwab callback URL to cache.
    """
    cache_data = {}
    
    if username:
        cache_data["TASTYTRADE_USERNAME"] = username
    if password:
        cache_data["TASTYTRADE_PASSWORD"] = password
    if fred_api_key:
        cache_data["FRED_API_KEY"] = fred_api_key
    if schwab_api_key:
        cache_data["SCHWAB_API_KEY"] = schwab_api_key
    if schwab_secret:
        cache_data["SCHWAB_SECRET"] = schwab_secret
    if schwab_callback_url:
        cache_data["SCHWAB_CALLBACK_URL"] = schwab_callback_url

    with open(CACHE_FILE, 'w') as file:
        json.dump(cache_data, file)
