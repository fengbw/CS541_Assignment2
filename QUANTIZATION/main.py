import nanopq
import numpy as np
import time
import sys
from dataReader import *

if __name__ == "__main__":
    # data = np.array(dataReader("../P53_train.ds"),dtype = np.float32)
    data = np.load("train.npy")
    # test = np.array(dataReader("../P53_test.ds"), dtype=np.float32)
    test = np.load("test.npy")
    dim = 5408

    # Instantiate with M=8 sub-spaces
    pq = nanopq.PQ(M=676)

    # Train codewords
    t1 = time.time()
    pq.fit(vecs=data[:8000], seed= 100)
    print(time.time()-t1)

    t2 = time.time()
    # Encode to PQ-codes
    X_code = pq.encode(data[::])
    print("Encode time is ", time.time()-t2)
    # Results: create a distance table online, and compute Asymmetric Distance to each PQ-code
    t3 = time.time()
    right = np.load("../groundtruth/linearScanResult3000.npy")
    res = []
    score = 0
    for i in range(np.shape(test)[0]):
        res = []
        dists = pq.dtable(test[i]).adist(X_code)
        res.append(list(np.argsort(dists)[:5]))
        if res[0][1] == right[i][1]:
            score += 1
        sys.stdout.write("\rTesting #{}, now score is {}".format(i,score))
    print()
    print("accuracy is ", score/np.shape(test)[0])
    print("query time is ", (time.time()-t3)/np.shape(test)[0])

    # # exact scan
    # distsExact = np.linalg.norm(data - query, axis=1) ** 2