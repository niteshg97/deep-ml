import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    """
    Perform linear regression using gradient descent.

    Args:
        X: Feature matrix of shape (m, n) where first column is all ones (for intercept)
        y: Target vector of shape (m,)
        alpha: Learning rate
        iterations: Number of gradient descent iterations
    
    Returns:
        Learned weights as a 1D array of shape (n,)
    """
    # Your code here: implement gradient descent
    m, n = X.shape 
    y = y.reshape(-1, 1)  # Ensure y is a column vector
    theta = np.zeros((n, 1))  # Initialize weights to zeros
    X = np.array(X)
    y = np.array(y)
    for i in range (iterations):
        prediction = X.dot(theta)
        gradiant = 1/m * (X.T).dot(prediction - y)
        theta  = theta - alpha * gradiant
    return theta.flatten()