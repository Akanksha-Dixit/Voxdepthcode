import cv2
import numpy as np
import glob

for f in glob.glob("/home/yashashwee/DepthFill/dataset/color/*"):
    img = cv2.imread(f,cv2.IMREAD_UNCHANGED)
    newFile = f.split("/")[-1].split(".")[0].zfill(6) + ".JPEG"
    print(newFile)
    cv2.imwrite("/home/yashashwee/DepthFill/dataset/MidAir/Kite_training/sunny/color_left/train/"+newFile,img)