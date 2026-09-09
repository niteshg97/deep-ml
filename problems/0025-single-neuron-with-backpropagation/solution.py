import torch
import torch.nn as nn

def train_neuron(features: torch.Tensor,labels: torch.Tensor,initial_weights: torch.Tensor,initial_bias: float,learning_rate: float,epochs: int) -> tuple[list[float], float, list[float]]:



    # Parameters that PyTorch should calculate gradients for
    w = initial_weights.clone().detach().requires_grad_(True)
    b = torch.tensor(
        initial_bias,
        dtype=torch.float32,
        requires_grad=True
    )

    mse_values = []

    for epoch in range(epochs):

        # -------------------------
        # 1. Forward pass
        # -------------------------
        z = torch.matmul(features, w) + b

        y_pred = torch.sigmoid(z)

        # MSE
        loss = nn.functional.mse_loss(y_pred, labels)

        # Record loss BEFORE updating weights
        mse_values.append(loss.item())

        # -------------------------
        # 2. Backpropagation
        # -------------------------
        loss.backward()

        # -------------------------
        # 3. Parameter update
        # -------------------------
        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad

        # -------------------------
        # 4. Clear gradients
        # -------------------------
        w.grad.zero_()
        b.grad.zero_()

    # Convert results
    updated_weights = [round(x, 4) for x in w.tolist()]
    updated_bias = round(b.item(), 4)
    mse_values = [round(x, 4) for x in mse_values]

    return updated_weights, updated_bias, mse_values
