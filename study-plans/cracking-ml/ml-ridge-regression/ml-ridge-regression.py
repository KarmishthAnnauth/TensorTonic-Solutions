import numpy as np

def ridge_regression(X: list, y: list, alpha: float, lr: float, epochs: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    N, D = X.shape
    weights = np.zeros(D, float)
    bias = 0.0
    for _ in range(epochs):
        predicitions = X @ weights + bias
        error = predicitions - y
        grad_w = 2 * (X.T @ error) / N + 2 * alpha * weights
        grad_b = 2 * np.mean(error)
        weights -= lr*grad_w
        bias -= lr*grad_b

    return [round(float(value), 4) for value in weights], round(float(bias), 4)
