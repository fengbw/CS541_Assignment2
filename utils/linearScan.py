import numpy as np
import dataReader


def linearScan(train,test,k=5):
    """ Get nearest neigbors
    Userr linear scan to find the nearest neighbors, using Euclidean distance

    Args:
        train: ndarry like traing data
        test: ndarry like query data
        k: how many nearest neighbors you want
    
    Returns:
        A list of nearset neighbors of each query
    """
    result = []
    for one in test:
        ext_test = np.tile(one,(train.shape[0],1))
        res = np.linalg.norm(train-ext_test,axis=1).argsort()
        result.append(res[:k])
    return result

