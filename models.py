import numpy as np
from scipy.optimize import minimize

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
