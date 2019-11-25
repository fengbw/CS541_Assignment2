import falconn
import timeit
import math
import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.getcwd())

from utils import *


if __name__ == "__main__":
    '''
    train_file="dataset/P53_train.ds"
    train = dataReader(train_file)
    train = np.array(train,dtype=np.float32)
    np.save("dataset/trainset/size"+str(len(dataset)),dataset)
    test_file="dataset/P53_test.ds"
    test = dataReader(test_file)
    test = np.array(test,dtype=np.float32)
    np.save("dataset/testset/size"+str(len(queries)),queries)

    '''
    train = np.load("dataset/trainset/size28059.npy")
    test = np.load("dataset/testset/size3000.npy")

    print(train.shape)



    # important parameters
    number_of_tables = 10
    k_neighbors=5
    number_of_probes=10
    number_of_function=10


    # falconn requires use float32
    assert train.dtype == np.float32
    assert test.dtype == np.float32
    
    #using the cosine similarity, normalize data
    train/= np.linalg.norm(train,axis=1).reshape(-1,1)
    test/= np.linalg.norm(test,axis=1).reshape(-1,1)
 
    #queries is test, dataset is train
    queries=test
    dataset=train



    print('Solving queries using linear scan')
    '''
    t1 = timeit.default_timer()
    answers=linearScan(dataset,queries,k_neighbors)
    np.save("groundtruth/linearScanResult"+str(len(queries)),answers)
    
    t2=timeit.default_timer()
    print("done")
    print('Linear scan time: {} per query'.format((t2 - t1) / float(len(queries))))
    '''
    '''
    answers=np.load("groundtruth/linearScanResult3000.npy")
    answers=answers[:len(queries)]
    answers=answers[:,:k_neighbors]
    '''
    #cosine distance
    answers = []
    for query in queries:
        answers.append(np.dot(dataset, query).argmax())

    
    print('Centering the dataset and queries')
    center = np.mean(dataset, axis=0)
    dataset -= center
    queries -= center
    print('Done')

    params_cp = falconn.LSHConstructionParameters()
    params_cp.dimension = len(dataset[0])
    params_cp.lsh_family=falconn.LSHFamily.CrossPolytope
    params_cp.distance_function=falconn.DistanceFunction.EuclideanSquared
    params_cp.l=number_of_tables

    params_cp.num_rotations=1
    params_cp.seed=5721840

    params_cp.num_setup_threads=0
    params_cp.storage_hash_table=falconn.StorageHashTable.BitPackedFlatHashTable

    falconn.compute_number_of_hash_functions(number_of_function,params_cp)

    print("contstruting the LSH table")
    t1=timeit.default_timer()
    table=falconn.LSHIndex(params_cp)
    table.setup(dataset)
    t2=timeit.default_timer()
    print("Done")
    print("Construction time: {}".format(t2-t1))


    query_object = table.construct_query_object()
    query_object.set_num_probes(number_of_probes)

    print("finding nearset neighbors")
    t1=timeit.default_timer()
    result=[]
    for query in queries:
        result.append(query_object.find_k_nearest_neighbors(query,k_neighbors))
    t2=timeit.default_timer()
    print("Done")
    print("per query time:{}".format((t2-t1)/len(result)))
    print("precision:",compareResult(answers,result))

'''
    # find the smallest number of probes to achieve accuracy 0.9
    # using the binary search
    print('Choosing number of probes')
    number_of_probes = number_of_tables

    def evaluate_number_of_probes(number_of_probes):
        query_object.set_num_probes(number_of_probes)
        score = 0
        for (i, query) in enumerate(queries):
            if answers[i] in query_object.get_candidates_with_duplicates(query):
                score += 1
        return float(score) / len(queries)

    while True:
        accuracy = evaluate_number_of_probes(number_of_probes)
        print('{} -> {}'.format(number_of_probes, accuracy))
        if accuracy >= 0.9:
            break
        number_of_probes = number_of_probes * 2
    if number_of_probes > number_of_tables:
        left = number_of_probes // 2
        right = number_of_probes
        while right - left > 1:
            number_of_probes = (left + right) // 2
            accuracy = evaluate_number_of_probes(number_of_probes)
            print('{} -> {}'.format(number_of_probes, accuracy))
            if accuracy >= 0.9:
                right = number_of_probes
            else:
                left = number_of_probes
        number_of_probes = right
    print('Done')
    print('{} probes'.format(number_of_probes))

    # final evaluation
    t1 = timeit.default_timer()
    score = 0
    for (i, query) in enumerate(queries):
        if query_object.find_nearest_neighbor(query) == answers[i]:
            score += 1
    t2 = timeit.default_timer()


    print('Query time: {}'.format((t2 - t1) / len(queries)))
    print('Precision: {}'.format(float(score) / len(queries)))
'''