import cv2
import glob
import numpy as np


fp = "/home/yashashwee/DepthFill/dataset/Corrected/"
# rp = "/home/yashashwee/DepthFill/dataset/LMSDepth/"
rp = "/home/yashashwee/cudaHello/outsNamed/"
# rp = "/home/yashashwee/DepthFill/dataset/holeFilled/"
# rp = "/home/yashashwee/DepthFill/dataset/DeepSmooth/"

# lp = "/home/yashashwee/DepthFill/dataset/LMSDepth/"
# rp2 = "/home/yashashwee/DepthFill/dataset/rectifiedDepthCompare/"
rtP = []
ltP = []
for f in glob.glob(fp+"*"):
    idx = f.split("/")[-1]
    gt = cv2.imread(f,cv2.IMREAD_UNCHANGED)
    rt = cv2.imread(rp+idx,cv2.IMREAD_UNCHANGED)

    gt = cv2.normalize(gt,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    rt = cv2.normalize(rt,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    print(idx,cv2.PSNR(gt,rt))
    rtP.append(cv2.PSNR(gt,rt))

print(np.mean(rtP))


