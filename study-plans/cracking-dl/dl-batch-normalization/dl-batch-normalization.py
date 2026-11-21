import numpy as np

def batch_norm(X: list, gamma: list, beta: list, running_mean: list, running_var: list, mode: str) -> dict:
    """
    Returns normalized output and resulting running statistics.
    """
    X = np.asarray(X, dtype = np.float64)
    gamma = np.asarray(gamma, dtype = np.float64)
    beta = np.asarray(beta, dtype = np.float64)
    running_mean=np.asarray(running_mean,dtype=np.float64).copy()
    running_var=np.asarray(running_var,dtype=np.float64).copy() 

    if mode == 'train':
        mean = np.mean(X, axis=0)
        variance = np.var(X, axis=0, ddof=0)
        output = gamma*(X-mean)/np.sqrt(variance+1e-5)+beta
        running_mean=0.9*running_mean+0.1*mean
        running_var=0.9*running_var+0.1*variance 

    else:
        output = gamma*(X-running_mean)/np.sqrt(running_var+1e-5)+beta
    clean=lambda value:[round(float(v),4) for v in value]
    return {"output":[clean(row) for row in output],"running_mean":clean(running_mean),"running_var":clean(running_var)}