import numpy as np

def gradient_descent(X, y, weights, learning_rate,
                     n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    for epoch in range(n_epochs):

        # Batch GD
        if method == 'batch':

            X_batch = X
            y_batch = y

            predictions = X_batch @ weights
            error = predictions - y_batch

            gradient = (2 / len(X_batch)) * (X_batch.T @ error)

            weights = weights - learning_rate * gradient

        # SGD 
        elif method == 'stochastic':

            for i in range(len(X)):

                X_batch = X[i:i+1]
                y_batch = y[i:i+1]

                predictions = X_batch @ weights
                error = predictions - y_batch

                gradient = (2 / len(X_batch)) * (X_batch.T @ error)

                weights = weights - learning_rate * gradient

        #  Mini-Batch GD 
        elif method == 'mini_batch':

            for i in range(0, len(X), batch_size):

                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]

                predictions = X_batch @ weights
                error = predictions - y_batch

                gradient = (2 / len(X_batch)) * (X_batch.T @ error)

                weights = weights - learning_rate * gradient

        else:
            raise ValueError("Invalid method")

    return weights