import cv2
import numpy as np
import pandas as pd
import glob

def diceCoeff(pred,truth):
    predFlat, trueFlat = [img.flatten() for img in [pred,truth]]
    intersection = np.sum(trueFlat*predFlat)
    return (2.0*intersection + 1)/(np.sum(trueFlat)+np.sum(predFlat)+1)

def prepare(img,value):
    return cv2.threshold(img,255.0*value,1,cv2.THRESH_BINARY)[1].astype(np.uint8)
    
def getImgPaths():
    imgPaths = []
    for f in glob.glob("train/*mask.tif"):
        imgPaths.append(f)
    return imgPaths
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

for csv in [test,train]:

    imgPaths = list(csv.img.values)
    thresholds = np.linspace(0.4,0.8,6)
    thresholdScores = list()

    for threshold in thresholds:
        print("*"*20)
        print(threshold)
        print("*"*20)
        nObs = len(imgPaths)
        scores = np.empty((nObs,2))
        for idx in range(nObs)[:]:
            f = imgPaths[idx]
            img,mask,fitted = cv2.imread(f.replace("_mask",""),0), cv2.imread(f,0), cv2.imread(f.replace("mask","fitted"),0)
            fittedBlur = cv2.GaussianBlur(fitted,(15,15),1)
            
            fittedPrep = prepare(fitted,threshold)
            fittedPrepBlur = prepare(fittedBlur,threshold)
            mask[mask==255] = 1

            score = diceCoeff(fittedPrep,mask)
            scoreBlur = diceCoeff(fittedPrepBlur,mask)
            scores[idx] = np.array([score,scoreBlur])
            def display():
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4)
                fig.tight_layout()
                fig.subplots_adjust(hspace=0)
                ax1.set_title(str(score))
                ax1.imshow(img,cmap=cm.gray)
                ax2.imshow(mask,cmap=cm.gray)
                ax2.set_title("Truth")
                ax3.imshow(fitted,cmap=cm.gray)
                ax3.set_title("Fitted")
                ax4.imshow(fittedPrep,cmap=cm.gray)
                ax3.set_title("Fitted Cleaned")
                plt.show()
            if idx % 250 == 0:
                print("%d out of %d" % (idx,nObs))
                pass
                #display()
        thresholdScores.append(scores)
        print("Mean score for threshold %f = %f, %f (blur)" % (threshold,scores.mean(0)[0],scores.mean(0)[1]))
    meanScores =[arr.mean(0) for arr in thresholdScores]

