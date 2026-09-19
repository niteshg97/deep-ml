import torch


def clip_grad_norm(parameters, max_norm: float) -> float:
    parameters = list(parameters)
    grads = []
    for p in parameters:
        if p.grad is not None:
            grads.append(p.grad)
    total = 0.0

    for g in grads:
        total += g.detach().norm().item() ** 2
    total_norm = total ** 0.5
    if total_norm <= max_norm:
        return total_norm

    scale = max_norm / total_norm
    for g in grads:
        g.mul_(scale)
    return total_norm