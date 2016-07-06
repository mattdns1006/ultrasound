import numpy as np
import cv2
import glob
import re


def prep(img):
    img = img.astype('float32')
    img = cv2.threshold(img, 128, 250, cv2.THRESH_BINARY)[1].astype(np.uint8)
    return img


def run_length_enc(label):
    from itertools import chain
    x = label.transpose().flatten()
    y = np.where(x > 0)[0]
    if len(y) < 10:  # consider as empty
        return ''
    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z+1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s+1, l+1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return ' '.join([str(r) for r in res])

def submission():

    rles = []
    ids = []

    total = 0
    for f in glob.glob("testPredictions/*"):
        number =re.findall("\d+",f)
        img = cv2.imread(f,0)
        img = prep(img)
        rle = run_length_enc(img)
        rles.append(rle)
        ids.append(int(number[0]))
        total += 1 
        if total % 100 == 0:
            print("{}/{}".format(total,5508))

    firstRow = "img,pixels"
    fileName = "submission.csv"
    with open(fileName,"w+") as f:
        f.write(firstRow + "\n")
        for i in range(total):
            s = str(ids[i]) + "," + rles[i]
            f.write(s + "\n")

if __name__ == "__main__":
    submission()
    print("Finished.")
