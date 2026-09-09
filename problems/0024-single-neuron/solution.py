import torch
import torch.nn.functional as F

def single_neuron_model(
    features: list[list[float]],
    labels: list[int],
    weights: list[float],
    bias: float
) -> tuple[list[float], float]:
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    W = torch.tensor(weights, dtype=torch.float32)
    b = torch.tensor(bias, dtype=torch.float32)
    z = torch.matmul(X, W) + b
    predictions = torch.sigmoid(z)
    mse = F.mse_loss(predictions, y)
    predictions = predictions.tolist()
    predictions = [round(p, 4) for p in predictions]

    mse = round(mse.item(), 4)

    return predictions, mse


    