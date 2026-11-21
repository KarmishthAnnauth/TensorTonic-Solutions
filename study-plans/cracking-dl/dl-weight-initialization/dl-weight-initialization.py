import math

def weight_init_params(layer_dims: list, method: str) -> list:
    """
    Returns initialization metadata for each weight matrix.
    """
    output = []
    for fan_in, fan_out in zip(layer_dims[:-1], layer_dims[1:]):
        if method == "random_normal":
            scale = 1.0
        elif method == "xavier_normal":
            scale = math.sqrt(2/(fan_in+fan_out))
        elif method == "xavier_uniform":
            scale = math.sqrt(6.0/(fan_in+fan_out))
        elif method == "kaiming_normal":
            scale = math.sqrt(2.0/fan_in)
        else:
            scale = math.sqrt(6.0/fan_in)

        output.append({"fan_in":fan_in,"fan_out":fan_out,"shape":[fan_out,fan_in],"scale":round(float(scale),4)})

    return output