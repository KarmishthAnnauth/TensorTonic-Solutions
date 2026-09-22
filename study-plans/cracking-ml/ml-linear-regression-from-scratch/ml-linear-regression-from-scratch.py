import numpy as np

def linear_regression_from_scratch(X: list, y: list, lr: float, epochs: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    N, D = X.shape
    weights = np.zeros(D, float)
    bias = 0.0

    for _ in range(epochs):
        error = (X @ weights + bias) - y
        weights -= lr*(2/len(y)) * (X.T @ error)
        bias -= lr*(2/len(y)) * np.sum(error)

    return (weights,bias)
