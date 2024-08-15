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

def save_cached_credentials(username, password, fred_api_key=None):
    """
    Save the provided username, password, and optionally FRED API Key to the cache file.

    Args:
        username (str): The username to cache.
        password (str): The password to cache.
        fred_api_key (str): The FRED API Key to cache.
    """
    cache_data = {"TASTYTRADE_USERNAME": username, "TASTYTRADE_PASSWORD": password}
    if fred_api_key:
        cache_data["FRED_API_KEY"] = fred_api_key

    with open(CACHE_FILE, 'w') as file:
        json.dump(cache_data, file)
