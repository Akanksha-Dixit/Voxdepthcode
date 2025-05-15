import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from scipy.stats import shapiro


# gtpath = "/home/yashashwee/DepthFill/dataset/Corrected/"
# # gtpath = "/home/yashashwee/DeepSmooth/evals/trajectory_0001/gt/"

# corpath = "/home/yashashwee/cudaHello/outsLN/"
# # corpath = "/home/yashashwee/cudaHello/dataset/LMSDepth/"
# # corpath = "/home/yashashwee/cudaHello/dataset/holeFilled/"
# # corpath = "/home/yashashwee/cudaHello/dataset/DeepSmooth/"

# occpath = "/home/yashashwee/cudaHello/dataset/CorrectedOcc/"

gtpath = "/home/yashashwee/cudaHello/dataset/MidAir/PLE_training/winter/gt_depth/trajectory_6000/"
# gtpath = "/home/yashashwee/DeepSmooth/evals/trajectory_0001/gt/"

imgsPath = "/home/yashashwee/cudaHello/outs6000/"
# corpath = gtpath
# corpath = imgsPath
# corpath = "/home/yashashwee/DeepSmooth/evals/trajectory_6000/pred/"
# corpath = "/home/yashashwee/cudaHello/dataset/LMSDepthMidair6000/"
corpath = "/home/yashashwee/cudaHello/dataset/RealsenseMidair6000/"


# corpath = "/home/yashashwee/DeepSmooth/evals/trajectory_0001/pred/"
occpath = "/home/yashashwee/cudaHello/dataset/MidAir/PLE_training/winter/stereo_occlusion/trajectory_6000/"

perList = []
for f in glob.glob(imgsPath+"*"):
    i = f.split("/")[-1].split(".")[0]
    filledI = i.zfill(6)
    gtImg = cv2.imread(gtpath+ filledI +".PNG",cv2.IMREAD_UNCHANGED).astype(np.float32) #[0].astype(np.uint16)
    # occImg = cv2.imread(occpath+filledI+".PNG",cv2.IMREAD_UNCHANGED)
    corimg = cv2.imread(corpath+ i +".PNG",cv2.IMREAD_UNCHANGED).astype(np.float32)
    
    gtImg[gtImg==0] = 1

    # print(np.min(gtImg),f)
    percError = (corimg-gtImg)/(gtImg)
    # print(percError)

    # print(np.min(percError))
    for i in percError:
        for j in i:
            perList.append(j)
    
    
    # if len(sqList)>300:
    #     break

np.random.shuffle(perList)
print(np.mean(perList))
print(np.std(perList))
print(shapiro(perList[:1000]))

# plt.hist(perList,bins=100)
# plt.show()