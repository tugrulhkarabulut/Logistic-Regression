import numpy as np

from standard_scale import standard_scale
from gradient_descent import gradient_descent

class LogisticRegression:
    def __init__(self, max_iter = 100, alpha = 0.01, tolerance = 0.001, C = 1, 
                                                                add_intercept = True):
        self.max_iter = 100
        self.C = C
        self.alpha = alpha
        self.tolerance = tolerance
        self.add_intercept = add_intercept
    
    def fit(self, X, y):
        X_values = self.__get_values(X)
        X_ = self.__preprocess_data(X_values)
        _, col_count = X_.shape
        y_ = self.__get_values(y)
        
        coef = np.ones( col_count )
        
        opt_coef, cost = gradient_descent(lambda c: self.__cost(X_, y_, c), 
                                          coef, alpha = self.alpha, tolerance = self.tolerance)
        
        self.coef_ = opt_coef
        self.cost_ = cost
        
        return opt_coef, cost
    
    def predict(self, X):
        X_values = self.__get_values(X)
        
        X_ = self.__preprocess_data(X_values)
        
        result = self.__sigmoid(X_, self.coef_)

        result[ result >= 0.5 ] = 1
        result[ result <= 0.5 ] = 0
        
        return result
        
            
    def __sigmoid(self, X, coef):
        return 1 / ( 1 + np.exp(-X @ coef) )
    
    def __cost(self, X, y, coef):
        
        h_x = self.__sigmoid(X, coef)
        
        # Handle undefined logarithm values
        h_x[ h_x == 0 ] +=  1e-6
        h_x[ h_x == 1 ] -= 1e-6
        
        row_count = X.shape[0]
        
        
        reg_matrix = np.zeros(coef.shape)
        reg_matrix[1:] = coef[1:]
        
        reg_term = ( 1/(2*row_count) ) * ( reg_matrix ** 2 )
        reg_term_sum = reg_term.sum()
        reg_term_sum = 0
        

        cost = self.C * ( -y * np.log(h_x) - (1 - y) * np.log(1 -  h_x) )
        avg_cost = cost.mean()
                        
        gradients = self.__gradient(X, y, coef)
                        
        return avg_cost + reg_term_sum, gradients
    
    def __gradient(self, X, y, coef):
        h_x = self.__sigmoid(X,  coef)

        reg_term = np.zeros(coef.shape)
        reg_term[1:] = coef[1:]
        
        row_count = X.shape[0]
        gradients = self.C * (X.T @ (h_x - y)) / row_count + reg_term / row_count
        return gradients
    
    def __get_values(self, X):
        if isinstance(X, np.ndarray):
            return X
        return np.array(X)
    def __preprocess_data(self, X):
        X_std = standard_scale(X)

        if self.add_intercept is True:
            X_std = self.__add_intercept(X_std)

        return X_std
    
    def __add_intercept(self, X):
        row_count, col_count = X.shape
        
        intercept = np.ones(row_count)
        X_values = np.zeros((row_count, col_count + 1))
        
        X_values[:, 0] = intercept
        X_values[:, 1:] = X
        
        return X_values