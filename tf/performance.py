import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class performance():
    def __init__(self):
        self.epochLosses = []
        self.epochScores = []
        self.losses = []
        self.scores = []
    def add(self,loss,score):
        self.losses.append(loss)
        self.scores.append(score)
    def endEpoch(self):
        self.epochLosses.append(np.array(self.losses).mean())
        self.epochScores.append(np.array(self.scores).mean())
        self.losses = []
        self.scores = []

def display(x,yPred,yTruth):
    imgs = np.vstack(tuple([row for row in x.squeeze()]))
    preds = np.vstack(tuple([row for row in yPred.squeeze()]))
    truth = np.vstack(tuple([row for row in yTruth.squeeze()]))
    predsTruth = np.hstack((preds,truth))
    plt.subplot(121)
    plt.imshow(imgs,cmap=cm.gray)
    plt.subplot(122)
    plt.imshow(predsTruth,cmap=cm.gray)
    plt.show()
