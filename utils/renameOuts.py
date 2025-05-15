import cv2
import numpy as np
import glob
import tqdm
import os

for f in tqdm.tqdm(glob.glob("outs0002/*")):
    img = cv2.imread(f,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,dsize=(640,360))
    newFile = str(int(f.split("/")[-1].split(".")[0])) + ".tif"
    # print(newFile)
    cv2.imwrite("outsNamed/"+newFile,img)