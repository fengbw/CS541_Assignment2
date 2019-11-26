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
    print(np.shape(data))
    dim = 5408
    # N, Nt, D = 10000, 2000, 128
    # X = np.random.random((N, D)).astype(np.float32)  # 10,000 128-dim vectors to be indexed
    # Xt = np.random.random((Nt, D)).astype(np.float32)  # 2,000 128-dim vectors for training
    query = np.random.random((dim,)).astype(np.float32)  # a query vector

    # Instantiate with M=8 sub-spaces
    pq = nanopq.PQ(M=4)

    # Train codewords
    t1 = time.time()
    pq.fit(vecs=data[::], seed= 1212)
    print(time.time()-t1)

    t2 = time.time()
    # Encode to PQ-codes
    X_code = pq.encode(data)  # (10000, 8) with dtype=np.uint8
    print("Encode time is ", time.time()-t2)
    # Results: create a distance table online, and compute Asymmetric Distance to each PQ-code
    t3 = time.time()
    right = np.load("../groundtruth/linearScanResult3000.npy")
    res = []
    score = 0
    for i in range(np.shape(test)[0]):
        res = []
        dists = pq.dtable(test[i]).adist(X_code)  # (10000, )
        res.append(list(np.argsort(dists)[:5]))
        for j in range(5):
            print(res)
            print(right)
            if res[j] == right[i][j]:
                score += 1
        sys.stdout.write("\r testing #{}, now score is {}".format(i,score))
    print(score)
    print("query time is ", (time.time()-t3)/np.shape(test)[0])


    # # exact scan
    # distsExact = np.linalg.norm(data - query, axis=1) ** 2