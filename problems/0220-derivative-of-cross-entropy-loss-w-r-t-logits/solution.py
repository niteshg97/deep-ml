import numpy as np

def cross_entropy_derivative(logits: list[float], target: int) -> list[float]:

    x = np.array(logits)
    exp_x = np.exp(x - np.max(x))
    softmax = exp_x / np.sum(exp_x)
    y = np.zeros(len(logits))
    y[target] = 1
    gradient = softmax - y

    return gradient.tolist()


