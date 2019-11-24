import nanopq
import numpy as np
import time

if __name__ == "__main__":
    N, Nt, D = 10000, 2000, 128
    X = np.random.random((N, D)).astype(np.float32)  # 10,000 128-dim vectors to be indexed
    Xt = np.random.random((Nt, D)).astype(np.float32)  # 2,000 128-dim vectors for training
    query = np.random.random((D,)).astype(np.float32)  # a 128-dim query vector

    # Instantiate with M=8 sub-spaces
    pq = nanopq.PQ(M=8)

    t1 = time.time()
    # Train codewords
    pq.fit(Xt)
    print(time.time()-t1)

    t2 = time.time()
    # Encode to PQ-codes
    X_code = pq.encode(X)  # (10000, 8) with dtype=np.uint8
    print("Encode time is ", time.time()-t2)

    # Results: create a distance table online, and compute Asymmetric Distance to each PQ-code
    t3 = time.time()
    dists = pq.dtable(query).adist(X_code)  # (10000, )
    print("query time is ",time.time()-t3)
