def standard_scale(X):
    return ( X - X.mean() ) / X.std()