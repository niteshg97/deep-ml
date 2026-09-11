import torch
import torch.nn as nn

def count_parameters(layers: list) -> int:
    total = 0

    for l in layers:

        if l["type"] == "dense":
            layer = nn.Linear(
                l["input_size"],
                l["output_size"],
                bias=l["use_bias"]
            )

        elif l["type"] == "conv2d":
            layer = nn.Conv2d(
                l["in_channels"],
                l["out_channels"],
                l["kernel_size"],
                bias=l["use_bias"]
            )

        elif l["type"] == "embedding":
            layer = nn.Embedding(
                l["num_embeddings"],
                l["embedding_dim"]
            )

        total += sum(p.numel() for p in layer.parameters())

    return total