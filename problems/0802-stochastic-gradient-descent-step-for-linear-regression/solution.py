import numpy as np

def sgd_update(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    learning_rate: float,
    n_iter: int
) -> list:
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    weights = np.array(weights, dtype=float)
    n_samples = X.shape[0]
    for i in range(n_iter):
        # Choose one sample
        idx = i % n_samples
        x_i = X[idx]
        y_i = y[idx]
        # Prediction
        pred = x_i @ weights
        # Error
        error = pred - y_i
        # Per-sample gradient
        gradient = 2 * error * x_i
        # Update weights
        weights = weights - learning_rate * gradient
    return weights.tolist()