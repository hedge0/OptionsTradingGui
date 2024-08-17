import numpy as np
from scipy.optimize import minimize
from math import log, sqrt, exp
from numba import njit

def svi_model(k, params):
    """
    SVI Model function.

    Args:
        k (float or array-like): Log-moneyness of the option.
        params (list): Parameters [a, b, rho, m, sigma] for the SVI model.

    Returns:
        float or array-like: The SVI model value for the given log-moneyness.
    """
    a, b, rho, m, sigma = params
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

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
                                Options are 'WLS', 'LS', 'RE', 'WRE'. Defaults to "WRE".

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
    elif method == "WRE":
        spread = y_ask - y_bid
        epsilon = 1e-8
        weights = 1 / (spread + epsilon)
        residuals = (model(k, params) - y_mid) / y_mid
        weighted_residuals = weights * residuals ** 2
        return np.sum(weighted_residuals)
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
    if model == svi_model:
        initial_guess = [0.01, 0.5, -0.3, 0.0, 0.2]
        bounds = [(0, 1), (0, 1), (-1, 1), (-1, 1), (0.01, 1)]
    else:
        initial_guess = [0.2, 0.3, 0.1, 0.2, 0.1]
        bounds = [(None, None), (None, None), (None, None), (None, None), (None, None)]
    
    result = minimize(objective_function, initial_guess, args=(k, y_mid, y_bid, y_ask, model, method), method='L-BFGS-B', bounds=bounds)
    return result.x

def compute_metrics(x, y_mid, model, params):
    """
    Compute the metrics for the model fit.

    Args:
        x (array-like): Strikes of the options.
        y_mid (array-like): Mid prices of the options.
        model (function): The volatility model used.
        params (list): The fitted parameters for the model.

    Returns:
        tuple: chi_squared and avE5 metrics.
    """
    k = np.log(x)
    y_fit = model(k, params)
    chi_squared = np.sum((y_mid - y_fit) ** 2)
    avE5 = np.mean(np.abs(y_mid - y_fit)) * 10000
    
    return chi_squared, avE5

@njit
def erf(x):
    """
    Compute the error function (erf) using a numerical approximation.

    Args:
        x (float): The value for which to compute the error function.

    Returns:
        float: The computed error function value.
    """
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x)

    return sign * y

@njit
def normal_cdf(x):
    """
    Compute the cumulative distribution function (CDF) of the standard normal distribution.

    Args:
        x (float): The value for which to compute the CDF.

    Returns:
        float: The computed CDF value.
    """
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

@njit
def barone_adesi_whaley_american_option_price(S, K, T, r, sigma, option_type='calls'):
    """
    Barone-Adesi Whaley approximation formula for American option pricing.
    
    Args:
        S: Current stock price.
        K: Strike price of the option.
        T: Time to expiration in years.
        r: Risk-free interest rate.
        sigma: Implied volatility.
        option_type: 'calls' or 'puts'.
    
    Returns:
        The calculated option price.
    """
    M = 2 * r / sigma**2
    n = 2 * (r - 0.5 * sigma**2) / sigma**2
    q2 = (-(n - 1) - sqrt((n - 1)**2 + 4 * M)) / 2
    
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    
    if option_type == 'calls':
        BAW = S * normal_cdf(d1) - K * exp(-r * T) * normal_cdf(d2)
        if q2 < 0:
            return BAW
        S_critical = K / (1 - 1/q2)
        if S >= S_critical:
            return S - K
        else:
            A2 = (S_critical - K) * (S_critical**-q2)
            return BAW + A2 * (S/S_critical)**q2
    elif option_type == 'puts':
        BAW = K * exp(-r * T) * normal_cdf(-d2) - S * normal_cdf(-d1)
        if q2 < 0:
            return BAW
        S_critical = K / (1 - 1/q2)
        if S <= S_critical:
            return K - S
        else:
            A2 = (K - S_critical) * (S_critical**-q2)
            return BAW + A2 * (S/S_critical)**q2
    else:
        raise ValueError("option_type must be 'calls' or 'puts'.")

def calculate_implied_volatility_baw(mid_price, S, K, r, T, option_type='calls', max_iterations=100, tolerance=1e-8):
    """
    Calculate implied volatility using the Barone-Adesi Whaley model.
    
    Args:
        mid_price: Observed option price (mid-price).
        S: Current stock price.
        K: Strike price of the option.
        r: Risk-free interest rate.
        T: Time to expiration in years.
        option_type: 'calls' or 'puts'.
        max_iterations: Maximum number of iterations for the bisection method.
        tolerance: Convergence tolerance.
    
    Returns:
        Implied volatility as a float.
    """
    lower_vol = 0.0001
    upper_vol = 2.0
    
    for i in range(max_iterations):
        mid_vol = (lower_vol + upper_vol) / 2
        price = barone_adesi_whaley_american_option_price(S, K, T, r, mid_vol, option_type)
        
        if abs(price - mid_price) < tolerance:
            return mid_vol
        
        if price > mid_price:
            upper_vol = mid_vol
        else:
            lower_vol = mid_vol
        
        if upper_vol - lower_vol < tolerance and upper_vol >= 2.0:
            upper_vol *= 2.0

    return mid_vol

def filter_strikes(x, S, num_stdev=1.25, two_sigma_move=False):
    """
    Filter strike prices around the underlying asset's price.

    Args:
        x: Array of strike prices.
        S: Current underlying price.
        num_stdev: Number of standard deviations for filtering (default 1.25).
        two_sigma_move: Adjust upper bound for a 2-sigma move (default False).

    Returns:
        Filtered array of strike prices within the specified range.
    """
    stdev = np.std(x)
    lower_bound = S - num_stdev * stdev
    upper_bound = S + num_stdev * stdev

    if two_sigma_move:
        upper_bound = S + 2 * stdev

    return x[(x >= lower_bound) & (x <= upper_bound)]
