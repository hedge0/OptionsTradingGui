import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import RBFInterpolator
from math import log, sqrt, exp
from numba import njit

def slv_model(k, params):
    """
    SLV Model function.

    Args:
        k (float or array-like): Log-moneyness of the option.
        params (list): Parameters [a, b, c, d, e] for the SLV model.

    Returns:
        float or array-like: The SLV model value for the given log-moneyness.
    """
    a, b, c, d, e = params
    return a + b * k + c * k**2 + d * k**3 + e * k**4

def rfv_model(k, params):
    """
    RFV Model function.

    Args:
        k (float or array-like): Log-moneyness of the option.
        params (list): Parameters [a, b, c, d, e] for the RFV model.

    Returns:
        float or array-like: The RFV model value for the given log-moneyness.
    """
    a, b, c, d, e = params
    return (a + b*k + c*k**2) / (1 + d*k + e*k**2)

def sabr_model(k, params):
    """
    SABR Model function.

    Args:
        k (float or array-like): Log-moneyness of the option.
        params (list): Parameters [alpha, beta, rho, nu, f0] for the SABR model.

    Returns:
        float or array-like: The SABR model value for the given log-moneyness.
    """
    alpha, beta, rho, nu, f0 = params
    return alpha * (1 + beta * k + rho * k**2 + nu * k**3 + f0 * k**4)

def rbf_model(k, y, epsilon=None, smoothing=0.0):
    """
    RBF Interpolation model function.

    Args:
        k (array-like): Log-moneyness of the option.
        y (array-like): Implied volatilities corresponding to log-moneyness.
        epsilon (float, optional): Regularization parameter for RBF. Defaults to None.
        smoothing (float, optional): Smoothing factor for RBF. Defaults to 0.0.

    Returns:
        function: A callable function that interpolates implied volatilities for given log-moneyness.
    """
    if epsilon is None:
        epsilon = np.mean(np.diff(np.sort(k)))
    rbf = RBFInterpolator(k[:, np.newaxis], y, kernel='multiquadric', epsilon=epsilon, smoothing=smoothing)
    return rbf

def objective_function(params, k, y_mid, y_bid, y_ask, model, method="WRE"):
    """
    Objective function to minimize during model fitting.

    Args:
        params (list): Model parameters.
        k (array-like): Log-moneyness of the options.
        y_mid (array-like): Mid prices of the options.
        y_bid (array-like): Bid prices of the options.
        y_ask (array-like): Ask prices of the options.
        model (function): The volatility model to be fitted.
        method (str, optional): The method for calculating the objective function.
                                Options are 'WLS', 'LS', 'RE'. 

    Returns:
        float: The calculated objective value to be minimized.
    """
    if method == "WLS":
        spread = y_ask - y_bid
        epsilon = 1e-8
        weights = 1 / (spread + epsilon)
        residuals = model(k, params) - y_mid
        weighted_residuals = weights * residuals ** 2
        return np.sum(weighted_residuals)
    elif method == "LS":
        residuals = model(k, params) - y_mid
        return np.sum(residuals ** 2)
    elif method == "RE":
        residuals = (model(k, params) - y_mid) / y_mid
        return np.sum(residuals ** 2)
    else:
        raise ValueError("Unknown method. Choose 'WLS', 'LS', 'RE', or 'WRE'.")

def fit_model(x, y_mid, y_bid, y_ask, model, method="WRE"):
    """
    Fit the chosen volatility model to the market data.

    Args:
        x (array-like): Strikes of the options.
        y_mid (array-like): Mid prices of the options.
        y_bid (array-like): Bid prices of the options.
        y_ask (array-like): Ask prices of the options.
        model (function): The volatility model to be fitted.
        method (str, optional): The method for the objective function. Defaults to "WRE".

    Returns:
        list: The fitted model parameters.
    """
    k = np.log(x)
    if model == rbf_model:
        rbf = rbf_model(k, y_mid)
        return rbf
    else:
        initial_guess = [0.2, 0.3, 0.1, 0.2, 0.1]
        bounds = [(None, None), (None, None), (None, None), (None, None), (None, None)]
    
    result = minimize(objective_function, initial_guess, args=(k, y_mid, y_bid, y_ask, model, method), method='L-BFGS-B', bounds=bounds)
    return result.x

def filter_strikes(x, S, num_stdev=1.25, two_sigma_move=False):
    """
    Filter strike prices around the underlying asset's price.

    Args:
        x (array-like): Array of strike prices.
        S (float): Current underlying price.
        num_stdev (float, optional): Number of standard deviations for filtering. Defaults to 1.25.
        two_sigma_move (bool, optional): Adjust upper bound for a 2-sigma move. Defaults to False.

    Returns:
        array-like: Filtered array of strike prices within the specified range.
    """
    stdev = np.std(x)
    lower_bound = S - num_stdev * stdev
    upper_bound = S + num_stdev * stdev

    if two_sigma_move:
        upper_bound = S + 2 * stdev

    return x[(x >= lower_bound) & (x <= upper_bound)]

@njit
def erf(x):
    """
    Approximation of the error function (erf) using a high-precision method.

    Parameters:
    - x (float): The input value.

    Returns:
    - float: The calculated error function value.
    """
    a1, a2, a3, a4, a5 = (
        0.254829592,
        -0.284496736,
        1.421413741,
        -1.453152027,
        1.061405429,
    )
    p = 0.3275911

    sign = 1 if x >= 0 else -1
    x = abs(x)
    
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * np.exp(-x * x))

    return sign * y

@njit
def normal_cdf(x):
    """
    Approximation of the cumulative distribution function (CDF) for a standard normal distribution.

    Parameters:
    - x (float): The input value.

    Returns:
    - float: The CDF value.
    """
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

@njit
def barone_adesi_whaley_american_option_price(S, K, T, r, sigma, q=0.0, option_type='calls'):
    """
    Calculate the price of an American option using the Barone-Adesi Whaley model with dividends.

    Args:
        S (float): Current stock price.
        K (float): Strike price of the option.
        T (float): Time to expiration in years.
        r (float): Risk-free interest rate.
        sigma (float): Implied volatility.
        q (float, optional): Continuous dividend yield. Defaults to 0.0.
        option_type (str, optional): Type of option ('calls' or 'puts'). Defaults to 'calls'.

    Returns:
        float: The calculated option price.
    """
    M = 2 * (r - q) / sigma**2
    n = 2 * (r - q - 0.5 * sigma**2) / sigma**2
    q2 = (-(n - 1) - sqrt((n - 1)**2 + 4 * M)) / 2
    
    d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    
    if option_type == 'calls':
        european_price = S * exp(-q * T) * normal_cdf(d1) - K * exp(-r * T) * normal_cdf(d2)
        if q >= r:
            return european_price
        if q2 < 0:
            return european_price
        S_critical = K / (1 - 1 / q2)
        if S >= S_critical:
            return S - K
        else:
            A2 = (S_critical - K) * (S_critical**-q2)
            return european_price + A2 * (S / S_critical)**q2
    
    elif option_type == 'puts':
        european_price = K * exp(-r * T) * normal_cdf(-d2) - S * exp(-q * T) * normal_cdf(-d1)
        if q >= r:
            return european_price
        if q2 < 0:
            return european_price
        S_critical = K / (1 + 1 / q2)
        if S <= S_critical:
            return K - S
        else:
            A2 = (K - S_critical) * (S_critical**-q2)
            return european_price + A2 * (S / S_critical)**q2
    
    else:
        raise ValueError("option_type must be 'calls' or 'puts'.")

@njit
def calculate_implied_volatility_baw(option_price, S, K, r, T, q=0.0, option_type='calls', max_iterations=100, tolerance=1e-8):
    """
    Calculate the implied volatility using the Barone-Adesi Whaley model with dividends.

    Parameters:
    - option_price (float): Observed option price (mid-price).
    - S (float): Current stock price.
    - K (float): Strike price of the option.
    - r (float): Risk-free interest rate.
    - T (float): Time to expiration in years.
    - q (float, optional): Continuous dividend yield. Defaults to 0.0.
    - option_type (str, optional): Type of option ('calls' or 'puts'). Defaults to 'calls'.
    - max_iterations (int, optional): Maximum number of iterations for the bisection method. Defaults to 100.
    - tolerance (float, optional): Convergence tolerance. Defaults to 1e-8.

    Returns:
    - float: The implied volatility.
    """
    lower_vol = 1e-5
    upper_vol = 5.0

    for i in range(max_iterations):
        mid_vol = (lower_vol + upper_vol) / 2
        price = barone_adesi_whaley_american_option_price(S, K, T, r, mid_vol, q, option_type)

        if abs(price - option_price) < tolerance:
            return mid_vol

        if price > option_price:
            upper_vol = mid_vol
        else:
            lower_vol = mid_vol

        if upper_vol - lower_vol < tolerance:
            break

    return mid_vol
