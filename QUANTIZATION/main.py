import nanopq
import numpy as np
import time
import sys
import os
sys.path.append(os.getcwd())

from utils import *
#from dataReader import *

if __name__ == "__main__":
    # data = np.array(dataReader("../P53_train.ds"),dtype = np.float32)
    data = np.load("dataset/trainset/raw_train.npy")[:28000]
    # test = np.array(dataReader("../P53_test.ds"), dtype=np.float32)
    test = np.load("dataset/testset/raw_test.npy")[:1500]
    dim = 5408
    data/= np.linalg.norm(data,axis=1).reshape(-1,1)
    test/= np.linalg.norm(test,axis=1).reshape(-1,1)
    # Instantiate with M=8 sub-spaces
    pq = nanopq.OPQ(M=208,Ks=512,verbose=False)

    # Train codewords
    t1 = time.time()
    pq.fit(vecs=data[:6000], seed= 100,)
    print(time.time()-t1)

    t2 = time.time()
    # Encode to PQ-codes
    X_code = pq.encode(data[::])
    print("Encode time is ", time.time()-t2)
    # Results: create a distance table online, and compute Asymmetric Distance to each PQ-code
    right = np.load("groundtruth/linearScanResult28000.npy")[:len(test)]

    t3 = time.time()
    #right = np.load("groundtruth/linearScanResultEuclidean2800_28000.npy")

    res = []
    for i in range(np.shape(test)[0]):
        dists = pq.dtable(test[i]).adist(X_code)
        res.append(list(np.argsort(dists)[:5]))
    print("query time is ", (time.time()-t3)/np.shape(test)[0])

    print("accuracy is ",compareResult(right,res))

    # # exact scan
    # distsExact = np.linalg.norm(data - query, axis=1) ** 2