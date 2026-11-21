import numpy as np

def backprop(X: list, y: list, W1: list, b1: list, W2: list, b2: list) -> dict:
    """
    Returns gradients for both weight matrices and bias vectors.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    W1 = np.asarray(W1, dtype=np.float64)
    W2 = np.asarray(W2, dtype=np.float64)
    b1=np.asarray(b1,dtype=np.float64)
    b2=np.asarray(b2,dtype=np.float64)

    z1 = X@W1+b1
    a1 = np.maximum(0.0, z1)
    prediction = a1@W2+b2
    dz2 = (2.0/len(X))*(prediction-y)
    dW2 = a1.T@dz2
    db2 = np.sum(dz2, axis=0)
    dz1 = (dz2@W2.T)*(z1>0.0)
    dW1 = X.T@dz1
    db1 = np.sum(dz1, axis=0)
    clean=lambda value:[[round(float(v),4) for v in row] for row in value]
    return {"dW1":clean(dW1),"db1":[round(float(v),4) for v in db1],"dW2":clean(dW2),"db2":[round(float(v),4) for v in db2]}
    