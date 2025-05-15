import cv2
import glob
import numpy as np


# fp = "/home/yashashwee/DepthFill/dataset/Corrected/"
fp = "/home/yashashwee/cudaHello/dataset/MidAir/PLE_training/winter/gt_depth/trajectory_6000/"
# rp = "/home/yashashwee/DepthFill/dataset/LMSDepth/"
rp = "/home/yashashwee/DeepSmooth/evals/trajectory_6000/pred/"
# rp = "/home/yashashwee/cudaHello/outs4000/"
# rp = "/home/yashashwee/DepthFill/dataset/holeFilled/"
# rp = "/home/yashashwee/DepthFill/dataset/DeepSmooth/"

# lp = "/home/yashashwee/DepthFill/dataset/LMSDepth/"
# rp2 = "/home/yashashwee/DepthFill/dataset/rectifiedDepthCompare/"
rtP = []
ltP = []
for f in glob.glob(rp+"*"):
    idx = f.split("/")[-1].split(".")[0].zfill(6)
    gt = cv2.imread(fp+idx+".PNG",cv2.IMREAD_UNCHANGED)
    rt = cv2.imread(f,cv2.IMREAD_UNCHANGED)
    gt = cv2.resize(gt,(640,480),None)

    # print(gt.shape,rt.shape)
    # print(np.max(gt),np.max(rt))
    gt = cv2.normalize(gt,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    rt = cv2.normalize(rt,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    print(idx,cv2.PSNR(gt,rt))
    rtP.append(cv2.PSNR(gt,rt))

print(np.mean(rtP))
