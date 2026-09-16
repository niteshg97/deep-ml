import torch


def activation_derivatives(x: float) -> dict[str, float]:
    """
    Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x
    using PyTorch autograd.
    """

    # Sigmoid
    x = torch.tensor(x, requires_grad=True)
    y = torch.sigmoid(x)
    y.backward()

    sigmoid_grad = x.grad.item()

    # Tanh
    x = torch.tensor(x, requires_grad=True)

    y = torch.tanh(x)
    y.backward()

    tanh_grad = x.grad.item()

    # ReLU
    x = torch.tensor(x, requires_grad=True)

    y = torch.relu(x)
    y.backward()

    relu_grad = x.grad.item()

    return {
        "sigmoid": sigmoid_grad,
        "tanh": tanh_grad,
        "relu": relu_grad
    }