import numpy as np

def gradient_descent(func, coef, alpha = 0.01, max_iter = 500, tolerance = 0.001):
    opt_coef = np.copy(coef)
    cost = 0
    for _ in range(max_iter):
        cost, gradient = func(opt_coef)
        opt_coef -= alpha * gradient
        
    return opt_coef, cost