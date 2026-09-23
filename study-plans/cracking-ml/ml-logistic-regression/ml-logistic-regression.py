import numpy as np

def logistic_regression(X: list, y: list, lr: float, n_iters: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    N, D = X.shape
    weights = np.zeros(D, float)
    bias = 0.0

    for _ in range(n_iters):
        logit = np.clip(X @ weights + bias, -500, 500)
        probabilities = 1 / (1 + np.exp(-logit))
        error = probabilities - y
        grad_w = (X.T @ error) / N
        grad_b = np.mean(error)
        weights -= lr*grad_w
        bias -= lr*grad_b

        
    return (weights, bias)
