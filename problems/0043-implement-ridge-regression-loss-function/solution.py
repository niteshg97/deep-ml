import numpy as np


def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
    n_samples = X.shape[0]
    predictions = X @ w
    mse = np.mean((y_true - predictions) ** 2)
    l2_penalty = alpha * np.sum(w**2)
    return mse + l2_penalty


def fit_ridge(X: np.ndarray, y_true: np.ndarray, alpha: float) -> np.ndarray:
    n_samples, n_features = X.shape
    A = X.T @ X + (n_samples * alpha) * np.eye(n_features)
    b = X.T @ y_true
    return np.linalg.solve(A, b)


class RidgeRegression:
    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.w = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.fit_intercept:
            ones = np.ones((X.shape[0], 1))
            X_prime = np.hstack([ones, X])
        else:
            X_prime = X

        w_prime = fit_ridge(X_prime, y, self.alpha)

        if self.fit_intercept:
            w_prime[0] = w_prime[0] - self.alpha * w_prime[0]

        self.w = w_prime

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.fit_intercept:
            ones = np.ones((X.shape[0], 1))
            X = np.hstack([ones, X])
        return X @ self.w
