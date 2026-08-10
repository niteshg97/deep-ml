import numpy as np

def entropy_and_cross_entropy(
    P: list[float],
    Q: list[float]
) -> tuple[float, float]:

    P = np.array(P, dtype=float)
    Q = np.array(Q, dtype=float)

    mask = P > 0

    entropy = -np.sum(
        P[mask] * np.log(P[mask])
    )

    cross_entropy = -np.sum(
        P[mask] * np.log(Q[mask])
    )

    return entropy, cross_entropy

	