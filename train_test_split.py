def train_test_split(X, ratio=0.8):
    split_size = int(X.shape[0] * ratio)
    X_train = X[:split_size, ...]
    X_test = X[split_size:, ...]

    return X_train, X_test
