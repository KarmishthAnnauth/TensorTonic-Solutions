import numpy as np

def forward_pass(x: list, weights: list, biases: list) -> dict:
    """
    Returns all activations and pre-activations.
    """
    activation = np.asarray(x, dtype=np.float64)
    activations = [activation.tolist()]
    pre_activations = []

    for index, (weight, bias) in enumerate(zip(weights, biases)):
        weight = np.asarray(weight, dtype = np.float64)
        bias = np.asarray(bias, dtype=np.float64)
        pre_activation = weight@activation+bias
        pre_activations.append(pre_activation.tolist())
        activation = pre_activation if index==len(weights)-1 else np.maximum(0.0, pre_activation)
        activations.append(activation.tolist())

    return {"activations":activations,"pre_activations":pre_activations}