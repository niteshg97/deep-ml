import numpy as np


def clip_gradients_by_global_norm(
    gradients: list[list[float]],
    max_norm: float
) -> list[list[float]]:
    """
    Clip gradients by global norm.
    """
    global_norm = np.sqrt(
        sum(np.sum(np.array(g) ** 2) for g in gradients)
    )
    if global_norm <= max_norm:
        return gradients
    factor = max_norm / global_norm
    ans = []

    for g in gradients:
        clipped = np.array(g) * factor
        ans.append(clipped.tolist())

    return ans


