import numpy as np
def activation_derivatives(x: float) -> dict[str, float]:
    """
    Compute derivatives of Sigmoid, Tanh, and ReLU at x.
    """
    sigmoid = 1 / (1 + np.exp(-x))
    val1 = sigmoid * (1 - sigmoid)

    tanh = np.tanh(x)
    val2 = 1 - tanh ** 2

    if x > 0:
        val3 = 1
    else:
        val3 = 0
		
    dicti = {
        "sigmoid": val1,
        "tanh": val2,
        "relu": val3
    }

    return dicti

	