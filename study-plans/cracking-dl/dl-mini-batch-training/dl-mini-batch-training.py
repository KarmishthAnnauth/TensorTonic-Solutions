import numpy as np

def mini_batch_training(X: list, y: list, weights: list, biases: list, lr: float, epochs: int, batch_size: int) -> list:
    """
    Returns the full-dataset loss after each epoch.
    """
    X = np.asarray(X, dtype=np.float64)
    y=np.asarray(y,dtype=np.float64)
    weights=[np.asarray(w,dtype=np.float64).copy() for w in weights]
    biases=[np.asarray(b,dtype=np.float64).copy() for b in biases]
    losses = []

    def forward(batch):
        activations = [batch]
        pre = []

        for index, (weight, bias) in enumerate(zip(weights, biases)):
            z = activations[-1]@weight.T+bias
            pre.append(z)
            activations.append(z if index==len(weights)-1 else np.maximum(0.0, z))
        return activations, pre

    for _ in range(epochs):
        for start in range(0, len(X), batch_size):
            batch_X = X[start:start+batch_size]
            batch_y = y[start:start+batch_size]
            activations, pre = forward(batch_X)
            delta = (activations[-1]-batch_y)/len(batch_X)
            grad_w = [None] * len(weights)
            grad_b = [None] * len(biases)

            for layer in range(len(weights)-1, -1, -1):
                grad_w[layer] = delta.T@activations[layer]
                grad_b[layer] = np.sum(delta, axis = 0)
                if layer > 0:
                    delta = (delta@weights[layer])*(pre[layer-1]>0.0)

            for layer in range(len(weights)):
                weights[layer] -= lr*grad_w[layer]
                biases[layer] -= lr*grad_b[layer]

        predictions = forward(X)[0][-1]
        losses.append(round(float(0.5*np.mean(np.sum((predictions-y)**2, axis=1))), 4))

    return losses
                