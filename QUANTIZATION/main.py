import nanopq
import numpy as np
import time
import heapq
from dataReader import *

if __name__ == "__main__":
    data = np.array(dataReader("../P53_test.ds"),dtype = np.float32)
    print(np.shape(data))
    dim = 5408
    # N, Nt, D = 10000, 2000, 128
    # X = np.random.random((N, D)).astype(np.float32)  # 10,000 128-dim vectors to be indexed
    # Xt = np.random.random((Nt, D)).astype(np.float32)  # 2,000 128-dim vectors for training
    query = np.random.random((dim,)).astype(np.float32)  # a query vector

    # Instantiate with M=8 sub-spaces
    pq = nanopq.PQ(M=8)

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
    res = []
    for i in range(5):
        dists = pq.dtable(data[i]).adist(X_code)  # (10000, )
        res.append(np.argsort(dists)[:5])
    print(res)

    # # exact scan
    # distsExact = np.linalg.norm(data - query, axis=1) ** 2