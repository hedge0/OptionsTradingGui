import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import RBFInterpolator
from numba import njit

@njit
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

@njit
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

@njit
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

def objective_function(params, k, y_mid, y_bid, y_ask, model, method="WLS"):
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
        raise ValueError("Unknown method. Choose 'WLS', 'LS', or 'RE'.")

def fit_model(x, y_mid, y_bid, y_ask, model, method="WLS"):
    """
    Fit the chosen volatility model to the market data.

    Args:
        x (array-like): Strikes of the options.
        y_mid (array-like): Mid prices of the options.
        y_bid (array-like): Bid prices of the options.
        y_ask (array-like): Ask prices of the options.
        model (function): The volatility model to be fitted.
        method (str, optional): The method for the objective function. Defaults to "WLS".

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