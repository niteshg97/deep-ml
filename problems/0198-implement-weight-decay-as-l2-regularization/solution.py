import numpy as np

def apply_weight_decay(
    parameters: list[list[float]],
    gradients: list[list[float]],
    lr: float,
    weight_decay: float,
    apply_to_all: list[bool]
) -> list[list[float]]:
    """
    Apply weight decay (L2 regularization) to parameters.
    """
    res = []

    for i in range(len(parameters)):
        w = np.array(parameters[i])
        g = np.array(gradients[i])
        w_new = w - lr * g # Gradient descent

        if apply_to_all[i]:
            w_new = w_new - lr * weight_decay * w

        res.append(w_new.tolist())

    return res