import numpy as np
def gini_impurity(y):
    y = np.array(y)

    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)

    return round(1 - np.sum(probabilities ** 2), 3)