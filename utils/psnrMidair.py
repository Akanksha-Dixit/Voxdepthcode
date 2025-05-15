import cv2
import glob
import numpy as np
import tqdm


# fp = "/home/yashashwee/DepthFill/dataset/Corrected/"
fp = "/home/yashashwee/cudaHello/dataset/MidAir/PLE_training/fall/gt_depth/trajectory_4000/"
# rp = "/home/yashashwee/DepthFill/dataset/LMSDepth/"
# rp = "/home/yashashwee/cudaHello/dataset/LMSDepthMidair4000/"
rp = "/home/yashashwee/cudaHello/outs/"
# rp = "/home/yashashwee/DepthFill/dataset/holeFilled/"
# rp = "/home/yashashwee/DepthFill/dataset/DeepSmooth/"
# rp = "/home/yashashwee/cudaHello/dataset/RealsenseMidair2/"
# lp = "/home/yashashwee/DepthFill/dataset/LMSDepth/"
# rp2 = "/home/yashashwee/DepthFill/dataset/rectifiedDepthCompare/"
rtP = []
ltP = []
for f in tqdm.tqdm(glob.glob(rp+"*")):
    idx = f.split("/")[-1].split(".")[0].zfill(6)
    gt = cv2.imread(fp+idx+".PNG",cv2.IMREAD_UNCHANGED)
    rt = cv2.imread(f,cv2.IMREAD_UNCHANGED)

    gt = cv2.normalize(gt,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    rt = cv2.normalize(rt,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    # print(idx,cv2.PSNR(gt,rt))
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