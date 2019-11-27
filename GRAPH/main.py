
import sys
sys.path.append("../utils")
from dataReader import *
from compareResult import *
import hnswlib
import numpy as np
import time

def hnsw():
    # test = dataReader("../P53_test.ds")
    # test = np.array(test, dtype = np.float32)
    # np.save("P53_test", test)
    # train = dataReader("../P53_train.ds")
    # train = np.array(train, dtype = np.float32)
    # np.save("P53_train", train)
    test = np.load("P53_test.npy")
    test = test[:300]
    test_labels = np.arange(len(test))
    train = np.load("P53_train.npy")
    train = train[:3000]
    train_labels = np.arange(len(train))
    train /= np.linalg.norm(train,axis=1).reshape(-1,1)
    test /= np.linalg.norm(test,axis=1).reshape(-1,1)
    result = np.load("../groundtruth/linearScanResult300.npy")
    result = result[:len(test)]
    result = result[:,:5]

    # items = [2, 12, 32, 48, 100]
    # for item in items:
    #     print("----this is attribuate {} ---".format(item))
    t1 = time.time()
    # Declaring index
    p = hnswlib.Index(space = 'l2', dim = len(train[0])) # possible options are l2, cosine or ip
    # Initing index - the maximum number of elements should be known beforehand
    p.init_index(max_elements = len(train), ef_construction = 200, M = 32)
    # Element insertion (can be called several times):
    p.add_items(train, train_labels)
    # Controlling the recall by setting ef:
    p.set_ef(100) # ef should always be > k
    t2 = time.time()
    print("index time is : {}".format(t2 - t1))

    # Query dataset, k - number of closest elements (returns 2 numpy arrays)
    labels, distances = p.knn_query(test, k = 5)
    t3 = time.time()
    print("query time is : {}".format((t3 - t2) / 3000))
    print("precision:",compareResult(result,labels))

if __name__ == "__main__":
    hnsw()
    # result = np.load("../groundtruth/linearScanResult3000.npy")
    # print(result[0][:5])
