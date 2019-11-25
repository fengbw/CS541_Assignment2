from dataReader import dataReader
import numpy as np

def npySave(wholesize:int,testsize:int):
    print("loading data")
    wholeData=dataReader(dataset/P53.ds)
    print("done")
    wholeData=np.array(wholeData,dtype=np.float32)
    wholeData=wholeData[:wholesize]
    testData=wholesize[:testsize]
    trainData=trainData[testsize:]

    np.save("dataset/trainset/size",trainData)
    np.save("dataset/testset/size",testData)

if __name__ == "__main__":
    npySave()