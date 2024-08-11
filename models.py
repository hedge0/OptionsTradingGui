import numpy as np
from scipy.optimize import minimize

def svi_model(k, params):
    a, b, rho, m, sigma = params
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

def slv_model(k, params):
    a, b, c, d, e = params
    return a + b * k + c * k**2 + d * k**3 + e * k**4

def rfv_model(k, params):
    a, b, c, d, e = params
    return (a + b*k + c*k**2) / (1 + d*k + e*k**2)

def sabr_model(k, params):
    alpha, beta, rho, nu, f0 = params
    return alpha * (1 + beta * k + rho * k**2 + nu * k**3 + f0 * k**4)

def objective_function(params, k, y, model):
    return np.sum((model(k, params) - y) ** 2)

def fit_model(x, y, model):
    k = np.log(x)
    if model == svi_model:
        initial_guess = [0.01, 0.5, -0.3, 0.0, 0.2]
        bounds = [(0, 1), (0, 1), (-1, 1), (-1, 1), (0.01, 1)]
    else:
        initial_guess = [0.2, 0.3, 0.1, 0.2, 0.1]
        bounds = [(None, None), (None, None), (None, None), (None, None), (None, None)]
    result = minimize(objective_function, initial_guess, args=(k, y, model), method='L-BFGS-B', bounds=bounds)
    return result.x

def compute_metrics(x, y, model, params):
    k = np.log(x)
    y_fit = model(k, params)
    
    # Chi-Squared Calculation
    chi_squared = np.sum((y - y_fit) ** 2)
    
    # Average Error (avE5) Calculation
    avE5 = np.mean(np.abs(y - y_fit)) * 10000
    
    return chi_squared, avE5
