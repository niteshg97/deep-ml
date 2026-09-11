def calculate_parameters(layers):

    total = 0

    for l in layers:

        if l["type"] == "dense":
            p = l["input_size"] * l["output_size"]

            if l.get("bias", True):
                p += l["output_size"]

        elif l["type"] == "conv2d":
            k = l["kernel_size"]
            p = l["in_channels"] * l["out_channels"] * k * k

            if l.get("bias", True):
                p += l["out_channels"]

        total += p

    return total
