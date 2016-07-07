import cv2
import numpy as np
import glob
from submission import prep
import pdb

def diceCoef(truth,pred):
    trueFlat = np.invert(truth.flatten())
    predFlat = np.invert(pred.flatten())
    intersection = np.sum(trueFlat * predFlat)
    return (2.0 * intersection)/(np.sum(trueFlat) + np.sum(predFlat))
    
def getImgPaths():
    imgPaths = []
    for f in glob.glob("train/*mask.tif"):
        imgPaths.append(f)
    return imgPaths

def calculateScore(f):
    truth, pred = cv2.imread(f,0), cv2.imread(f.replace("mask","fitted"),0)
    truth, pred = [cv2.threshold(img, int(256*0.512), 1, cv2.THRESH_BINARY)[1].astype(np.uint8) for img in [truth,pred]]
    score = diceCoef(truth,pred)
    return score


if __name__ == "__main__":
    imgPaths = getImgPaths()
    scores = np.empty(len(imgPaths))
    i =0 
    for f in imgPaths:
        scores[i] = calculateScore(f)
        i+=1
        if i % 1000 ==0:
            print(i)




