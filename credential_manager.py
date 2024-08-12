import os
import json

CACHE_FILE = 'credentials_cache.json'

def load_cached_credentials():
    """
    Load cached credentials from a file.

    Returns:
        dict: A dictionary with the cached username and password if the file exists, otherwise an empty dictionary.
    """
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as file:
            return json.load(file)
    return {}

def save_cached_credentials(username, password):
    """
    Save the provided username and password to the cache file.

    Args:
        username (str): The username to cache.
        password (str): The password to cache.
    """
    with open(CACHE_FILE, 'w') as file:
        json.dump({"TASTYTRADE_USERNAME": username, "TASTYTRADE_PASSWORD": password}, file)
