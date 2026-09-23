import numpy as np

def lasso_regression(X: list, y: list, alpha: float, lr: float, epochs: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    N, D = X.shape
    weights = np.zeros(D, float)
    bias = 0.0

    for _ in range(epochs):
        predictions = X @ weights + bias
        error = predictions - y
        weight_grad = (2/N) * (X.T @ error) + alpha * np.sign(weights)
        bias_grad = 2 * np.mean(error)
        weights -= lr*weight_grad
        bias -= lr*bias_grad

    return [round(float(value), 4) for value in weights], round(float(bias), 4)
