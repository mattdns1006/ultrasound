import cv2
import numpy as np

def diceCoef(truth,pred):
    smooth = 1
    trueFlat = np.flatten(truth)
    predFlat = np.flatten(pred)
    intersection = np.sum(trueFlat * predFlat)
    return (2.0 * intersection + smooth)/(np.sum(trueFlat) + np.sum(predFlat) + smooth)
    
