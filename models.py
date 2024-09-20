import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import RBFInterpolator
import QuantLib as ql

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

def calculate_option_price_quantlib(S, K, r, T, q, sigma, option_type='calls', american=False):
    """
    Calculate the option price using QuantLib.

    Args:
        S (float): Current stock price.
        K (float): Strike price of the option.
        r (float): Risk-free interest rate.
        T (float): Time to expiration in years.
        q (float): Continuous dividend yield.
        sigma (float): Implied volatility.
        option_type (str, optional): Type of option ('calls' or 'puts'). Defaults to 'calls'.
        american (bool, optional): If True, use American exercise, otherwise use European. Defaults to False.

    Returns:
        float: The option price (NPV).
    """
    spot_price = ql.SimpleQuote(S)
    strike_price = K
    risk_free_rate = ql.SimpleQuote(r)
    dividend_yield = ql.SimpleQuote(q)
    volatility = ql.SimpleQuote(sigma)

    expiration_date = ql.Date().todaysDate() + int(T * 365)
    payoff = ql.PlainVanillaPayoff(ql.Option.Call if option_type == 'calls' else ql.Option.Put, strike_price)

    if american:
        exercise = ql.AmericanExercise(ql.Date().todaysDate(), expiration_date)
    else:
        exercise = ql.EuropeanExercise(expiration_date)

    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(spot_price),
        ql.YieldTermStructureHandle(ql.FlatForward(0, ql.NullCalendar(), ql.QuoteHandle(dividend_yield), ql.Actual365Fixed())),
        ql.YieldTermStructureHandle(ql.FlatForward(0, ql.NullCalendar(), ql.QuoteHandle(risk_free_rate), ql.Actual365Fixed())),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(0, ql.NullCalendar(), ql.QuoteHandle(volatility), ql.Actual365Fixed()))
    )

    option = ql.VanillaOption(payoff, exercise)

    if american:
        option.setPricingEngine(ql.BaroneAdesiWhaleyApproximationEngine(process))
    else:
        option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

    return option.NPV()

def calculate_iv_quantlib(option_price, S, K, r, T, q, option_type='calls', upper_bound=10.0, retries=0, max_retries=5):
    """
    Calculate the implied volatility using the Barone-Adesi Whaley model with dividends.
    Args:
        option_price (float): Observed option price (mid-price).
        S (float): Current stock price.
        K (float): Strike price of the option.
        r (float): Risk-free interest rate.
        T (float): Time to expiration in years.
        q (float, optional): Continuous dividend yield. Defaults to 0.0.
        option_type (str, optional): Type of option ('calls' or 'puts'). Defaults to 'calls'.
        upper_bound (float, optional): Upper bound for volatility search range. Defaults to 1.0.
        retries (int, optional): Number of retries attempted. Defaults to 0.
        max_retries (int, optional): Maximum number of retries allowed. Defaults to 5.

    Returns:
        float: The implied volatility.
    """
    spot_price = ql.SimpleQuote(S)
    strike_price = K
    risk_free_rate = ql.SimpleQuote(r)
    dividend_yield = ql.SimpleQuote(q)
    volatility = ql.SimpleQuote(0.2)

    expiration_date = ql.Date().todaysDate() + int(T * 365)
    payoff = ql.PlainVanillaPayoff(ql.Option.Call if option_type == 'calls' else ql.Option.Put, strike_price)
    exercise = ql.AmericanExercise(ql.Date().todaysDate(), expiration_date)

    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(spot_price),
        ql.YieldTermStructureHandle(ql.FlatForward(0, ql.NullCalendar(), ql.QuoteHandle(dividend_yield), ql.Actual365Fixed())),
        ql.YieldTermStructureHandle(ql.FlatForward(0, ql.NullCalendar(), ql.QuoteHandle(risk_free_rate), ql.Actual365Fixed())),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(0, ql.NullCalendar(), ql.QuoteHandle(volatility), ql.Actual365Fixed()))
    )

    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(ql.BaroneAdesiWhaleyApproximationEngine(process))

    try:
        iv = option.impliedVolatility(option_price, process, 1e-8, 100, 0.2, upper_bound)
        return iv
    except RuntimeError as e:
        if 'root not bracketed' in str(e):
            # Incrementally increase the upper bound and retry, but cap the retries to avoid infinite recursion.
            if retries < max_retries:
                return calculate_iv_quantlib(option_price, S, K, r, T, q, option_type, upper_bound * 2, retries + 1, max_retries)
            else:
                raise RuntimeError(f"Max retries reached: Implied volatility could not be found after {max_retries} attempts.")
        else:
            raise e
