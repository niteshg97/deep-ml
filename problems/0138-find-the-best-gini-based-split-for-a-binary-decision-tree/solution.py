import numpy as np
from typing import Tuple

def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """Return the (feature_index, threshold) that minimises weighted Gini impurity."""
    n_samples, n_features = X.shape
    best_gini = float("inf")
    best_feature = -1
    best_threshold = None
    def calculate_gini(labels): #helper to calculate gini 
        if len(labels) == 0:
            return 0.0
        _, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    for feature_index in range(n_features):
        feature_values = X[:, feature_index]
        thresholds = np.unique(feature_values)
        for threshold in thresholds:
            left_mask = feature_values <= threshold
            right_mask = feature_values > threshold

            y_left = y[left_mask]
            y_right = y[right_mask]
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            left_gini = calculate_gini(y_left)
            right_gini = calculate_gini(y_right)
            # Calculate weighted Gini
            n_left = len(y_left)
            n_right = len(y_right)

            weighted_gini = (
                (n_left / n_samples) * left_gini +(n_right / n_samples) * right_gini)
            # Keep the best split
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature_index
                best_threshold = threshold

    return best_feature, best_threshold