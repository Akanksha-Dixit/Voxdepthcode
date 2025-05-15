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


# for g in glob.glob(fp):
#     filename = g.split("/")[-1]
#     gt = cv2.imread(g,cv2.IMREAD_UNCHANGED)
#     rt = cv2.imread(rp+filename,cv2.IMREAD_UNCHANGED)
#     lt = cv2.imread(lp+filename,cv2.IMREAD_UNCHANGED)

#     gt = cv2.normalize(gt,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
#     rt = cv2.normalize(rt,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
#     lt = cv2.normalize(lt,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)

#     rtP.append(cv2.PSNR(gt,rt))
#     ltP.append(cv2.PSNR(gt,lt))
#     print(filename)
#     print("Rect ",cv2.PSNR(gt,rt))
#     print("LMS ",cv2.PSNR(gt,lt))
    
# print("Rectified Avg PSNR: ",np.average(rtP))
# print("LMS average PSNR: ",np.average(ltP))