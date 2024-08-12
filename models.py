import numpy as np
from scipy.optimize import minimize

# SVI Model
def svi_model(k, params):
    a, b, rho, m, sigma = params
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

# SLV Model
def slv_model(k, params):
    a, b, c, d, e = params
    return a + b * k + c * k**2 + d * k**3 + e * k**4

# RFV Model
def rfv_model(k, params):
    a, b, c, d, e = params
    return (a + b*k + c*k**2) / (1 + d*k + e*k**2)

# SABR Model
def sabr_model(k, params):
    alpha, beta, rho, nu, f0 = params
    return alpha * (1 + beta * k + rho * k**2 + nu * k**3 + f0 * k**4)

# Objective Function for WLS, LS, and RE
def objective_function(params, k, y_mid, y_bid, y_ask, model, method="WLS"):
    if method == "WLS":
        # Calculate spreads
        spread = y_ask - y_bid
        # Avoid division by zero by adding a small epsilon to spread
        epsilon = 1e-8
        weights = 1 / (spread + epsilon)
        # Calculate weighted least squares
        residuals = model(k, params) - y_mid
        weighted_residuals = weights * residuals ** 2
        return np.sum(weighted_residuals)
    elif method == "LS":
        # Calculate least squares
        residuals = model(k, params) - y_mid
        return np.sum(residuals ** 2)
    elif method == "RE":
        # Calculate relative error
        residuals = (model(k, params) - y_mid) / y_mid
        return np.sum(residuals ** 2)
    else:
        raise ValueError("Unknown method. Choose 'WLS', 'LS', or 'RE'.")

# Model Fitting Function
def fit_model(x, y_mid, y_bid, y_ask, model, method="WLS"):
    k = np.log(x)
    if model == svi_model:
        initial_guess = [0.01, 0.5, -0.3, 0.0, 0.2]
        bounds = [(0, 1), (0, 1), (-1, 1), (-1, 1), (0.01, 1)]
    else:
        initial_guess = [0.2, 0.3, 0.1, 0.2, 0.1]
        bounds = [(None, None), (None, None), (None, None), (None, None), (None, None)]
    
    result = minimize(objective_function, initial_guess, args=(k, y_mid, y_bid, y_ask, model, method), method='L-BFGS-B', bounds=bounds)
    return result.x

# Metrics Computation Function
def compute_metrics(x, y_mid, model, params):
    k = np.log(x)
    y_fit = model(k, params)
    
    # Chi-Squared Calculation
    chi_squared = np.sum((y_mid - y_fit) ** 2)
    
    # Average Error (avE5) Calculation
    avE5 = np.mean(np.abs(y_mid - y_fit)) * 10000
    
    return chi_squared, avE5
