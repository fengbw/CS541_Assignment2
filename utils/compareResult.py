import numpy as np 

def compareResult(real,test):
    """calculate the precision of the ANN result

    Args:
        real: array like real result, can be 1-D or 2-D
        test: array like query result, can be 1-D or 2-D

    Returns:
        Decimal value of precision, can be regarded as recall rate
    """

    lens=len(test[0])
    res=0
    real=np.array(real)
    test=np.array(test)
    amount=real.size


    for i in range(len(real)):
        temp=np.intersect1d(real[i],test[i]).size
        res+=temp

    return float(res)/amount
