import numpy as np
import numpy.random as rng
import glob
import cv2
import os

class dataGenerator():
    def __init__(self,cvSplit=0.8,batchSize=5,inputDim=(96,128),outputDim=(12,16)):
        rng.seed(1006)
        self.trainPaths, self.testPaths = [glob.glob(s+"/*[0-9].tif") for s in ["train","test"]]
        self.batchSize = batchSize
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.originalSize = (420,580)
        print("%d train paths and %d test paths" % (len(self.trainPaths),len(self.testPaths)))
        
        # Split train into CV and non CV (cross validation)
        rng.shuffle(self.trainPaths)
        cvSplitPoint = int(cvSplit*len(self.trainPaths))
        self.trainPathsCV, self.testPathsCV = self.trainPaths[:cvSplitPoint], self.trainPaths[cvSplitPoint:]
        assert len(set(self.trainPathsCV).intersection(set(self.testPathsCV))) == 0 
        
        #self.trainPathsCV = self.trainPathsCV[:500]
        #self.testPathsCV = self.testPathsCV[:100]
        
        print("Train set split into %d train CV paths and %d test CV paths" % (len(self.trainPathsCV),len(self.testPathsCV)))     


    def loadImg(self,path,CV,augment=0,method=cv2.INTER_CUBIC):
        img = cv2.imread(path,0)
        maskPath = path.replace(".tif","_mask.tif")
        if os.path.exists(path = maskPath):
            mask = cv2.imread(maskPath,0)
            maskOrig = mask.copy()
        
        if augment == 1 and CV == 1:
            rows,cols = img.shape
            
            M = cv2.getRotationMatrix2D((cols/2,rows/2),np.random.uniform(-5,5),1)
            tX, tY = np.random.randint(0,10,2)
            M[0,2] = tX
            M[1,2] = tY
            img,mask = [im[5:rows-5, 5:cols-5] for im in [img,mask]]
            img,mask = [cv2.warpAffine(im,M,(cols,rows),borderMode = 1) for im in [img,mask]]
            maskOrig = mask.copy()        
            img = cv2.resize(img,self.inputDim, interpolation = method)
            mask = cv2.resize(mask,self.outputDim, interpolation = method)
            
            return img,mask, maskOrig
        elif augment == 0 and CV == 1:
            img = cv2.resize(img,self.inputDim, interpolation = method)
            mask = cv2.resize(mask,self.outputDim, interpolation = method)
            return img,mask, maskOrig
        elif train == 0:
            img = cv2.resize(img,self.inputDim, interpolation = method)
            return img, _, _
        
    def gen(self,train):
        if train==1:
            paths = self.trainPathsCV
            rng.shuffle(paths)
            nObs = len(paths)
            augment = 1
            print("Augmenting")
            print("Training paths length =  %d" % nObs)
        elif train == 0:
            paths = self.testPathsCV
            rng.shuffle(paths)
            nObs = len(paths)
            augment = 0
            print("Not augmenting")
            print("Testing paths length =  %d" % nObs)
        self.idx = 0
        finished = 0
        while True:
            batchX = np.empty((self.batchSize,self.inputDim[0],self.inputDim[1],1))
            batchY = np.empty((self.batchSize,self.outputDim[0],self.outputDim[1],1))
            batchYOrig = np.empty((self.batchSize,self.originalSize[0],self.originalSize[1],1))
            idx = 0
            for i in range(self.idx,min(self.batchSize+self.idx,nObs)):
                x,y,yOrig = self.loadImg(paths[i],CV=1,augment=augment)
                x=x/255.0
                y=y/255.0
                yOrig=yOrig/255.0
                
                x.resize(self.inputDim[0],self.inputDim[1],1), y.resize(self.outputDim[0],self.outputDim[1],1), yOrig.resize(self.originalSize[0],self.originalSize[1],1)
                batchX[idx],batchY[idx],batchYOrig[idx] = x, y, yOrig
                idx += 1
            self.idx += self.batchSize
            if self.idx >= nObs:
                self.idx = 0
                finished = 1
            yield batchX,batchY,batchYOrig, finished 
