from dataReader import *
import hnswlib
import numpy as np
import time

def hnsw():
    dataset = dataReader("../P53_test.ds")
    data = np.array(dataset, dtype = np.float32)
    print(data.shape)
    data_labels = np.arange(len(dataset))
    # Declaring index
    p = hnswlib.Index(space = 'l2', dim = len(dataset[0])) # possible options are l2, cosine or ip

    # Initing index - the maximum number of elements should be known beforehand
    time1 = time.time()
    p.init_index(max_elements = len(dataset), ef_construction = 200, M = 32)

    # Element insertion (can be called several times):
    p.add_items(data, data_labels)

    # Controlling the recall by setting ef:
    p.set_ef(50) # ef should always be > k
    time2 = time.time()
    print("index time is :", time2 - time1)

    # Query dataset, k - number of closest elements (returns 2 numpy arrays)
    labels, distances = p.knn_query(data[:5], k = 5)
    time3 = time.time()
    print("query time is :", time3 - time2)

    print(labels)
    print(distances)



if __name__ == "__main__":
    hnsw()
