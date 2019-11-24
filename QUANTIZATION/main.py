import nanopq
import numpy as np
import time
import pandas as pd

if __name__ == "__main__":
    number_of_queries = 1000
    number_of_tables = 50

    dataset_file = '../game_dataset.csv'
    df_data = pd.read_csv(dataset_file, index_col=0)
    df_data = df_data.astype(np.float32)
    df_data = df_data[:9999]
    np_data = df_data[['x', 'y']].values

    # falconn requires use float32
    assert np_data.dtype == np.float32

    # using the cosine similarity, normalize data
    np_data /= np.linalg.norm(np_data, axis=1).reshape(-1, 1)

    queries = np.array(np_data[len(np_data) - number_of_queries:])
    dataset = np.array(np_data[:len(np_data) - number_of_queries])
    # N, Nt, D = 10000, 2000, 128
    # X = np.random.random((N, D)).astype(np.float32)  # 10,000 128-dim vectors to be indexed
    # Xt = np.random.random((Nt, D)).astype(np.float32)  # 2,000 128-dim vectors for training
    # query = np.random.random((D,)).astype(np.float32)  # a 128-dim query vector

    # Instantiate with M=8 sub-spaces
    pq = nanopq.PQ(M=2)

    t1 = time.time()
    # Train codewords
    pq.fit(np_data)
    print(time.time()-t1)

    t2 = time.time()
    # Encode to PQ-codes
    X_code = pq.encode(np_data)  # (10000, 8) with dtype=np.uint8
    print("Encode time is ", time.time()-t2)

    # Results: create a distance table online, and compute Asymmetric Distance to each PQ-code
    t3 = time.time()
    dists = pq.dtable(queries[0]).adist(X_code)  # (10000, )
    print("query time is ",time.time()-t3)
