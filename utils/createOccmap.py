import cv2
import numpy as np
import glob

corpath = "/home/yashashwee/cudaHello/dataset/tempCor/"
basepath = "/home/yashashwee/cudaHello/dataset/baseNonZero/"
outpath = "/home/yashashwee/cudaHello/dataset/RealsenseOcc/"

white = np.iinfo(np.uint16).max
for f in glob.glob(corpath+"*"):
    idx = f.split("/")[-1]
    
    gtimg = cv2.imread(f,cv2.IMREAD_UNCHANGED)
    baseimg = cv2.imread(basepath+idx,cv2.IMREAD_UNCHANGED)

    outimg = np.ones_like(baseimg)*white
    outimg[np.logical_and(baseimg==white,gtimg!=white)] = 0

    cv2.imwrite(outpath+idx,outimg)