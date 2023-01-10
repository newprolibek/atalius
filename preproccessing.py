import numpy as np

def batches(X, y, batch_size):

    if(batch_size == None):
        batch_size = len(X)
    
    assert len(X) == len(y), "Length of X and y should be equal"

    X = np.array(X)
    y = np.array(y)
    perm = np.random.permutation(len(X))

    for i in range(len(X)//batch_size):
        start = i*batch_size
        end = i*batch_size + batch_size

        yield X[perm[start:end]], y[perm[start:end]]