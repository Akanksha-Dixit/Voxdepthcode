import cv2
import numpy as np
import glob
import tqdm

gtpath = "/home/yashashwee/DepthFill/dataset/Corrected/"
# gtpath = "/home/yashashwee/DeepSmooth/evals/trajectory_0001/gt/"

# corpath = "/home/yashashwee/cudaHello/outsMB/"
# corpath = "/home/yashashwee/cudaHello/dataset/LMSDepth/"
# corpath = "/home/yashashwee/cudaHello/dataset/holeFilled/"
corpath = "/home/yashashwee/cudaHello/dataset/DeepSmooth/"

occpath = "/home/yashashwee/cudaHello/dataset/CorrectedOcc/"

white = np.iinfo(np.uint16).max
sqList = []
for f in tqdm.tqdm(glob.glob(gtpath+"*")):
    i = f.split("/")[-1].split(".")[0]
    num = int(i)-6
    filledI = i.zfill(6)
    gtImg = cv2.imread(gtpath+ i +".tif",cv2.IMREAD_UNCHANGED)
    occImg = cv2.imread(occpath+ i +".tif",cv2.IMREAD_UNCHANGED)
    corimg = cv2.imread(corpath+i+".tif",cv2.IMREAD_UNCHANGED) #.astype(np.uint16)
    # corimg = cv2.normalize(corimg,None,0,white,cv2.NORM_MINMAX)
    # print(gtImg.shape,occImg.shape,corimg.shape)
    # print(np.max(gtImg),np.max(occImg),np.max(corimg))
    if len(corimg[occImg==0])==0:
        continue
    rmse = np.mean(np.square(corimg[occImg==0]-gtImg[occImg==0]))
    sqList.append(rmse)


print("RMSE : ",np.sqrt(np.mean(sqList)))
