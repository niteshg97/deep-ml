import numpy as np

def compute_cross_entropy_loss(
    predicted_probs: np.ndarray,
    true_labels: np.ndarray,
    epsilon=1e-15
) -> float:
    predicted_probs = np.array(predicted_probs, dtype=float)
    true_labels = np.array(true_labels, dtype=float)
    # Numerical stability
    predicted_probs = np.clip(
        predicted_probs,
        epsilon,
        1 - epsilon
    )
    # Cross entropy
    loss = -np.sum(
        true_labels * np.log(predicted_probs),
        axis=1
    )
    # Average across samples
    return np.mean(loss)

