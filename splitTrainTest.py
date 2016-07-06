import numpy as np
import cv2
import glob
import pandas as pd

trainPath = "train/"
np.random.seed(1)

allFiles = []
for f in glob.glob(trainPath+"*mask*"):
    allFiles.append(f)

perm = np.random.permutation(len(allFiles))
split = int(perm.size*0.8)
trainIdx, testIdx = perm[:split],perm[split:]
train = [allFiles[f] for f in trainIdx]
test = [allFiles[f] for f in testIdx]

print("Intersection of train and test",[i for i in train if i in test])

trainDf = pd.DataFrame(train,columns=["img"])
testDf = pd.DataFrame(test,columns=["img"])

trainDf.to_csv("train.csv",index=0)
testDf.to_csv("test.csv",index=0)






