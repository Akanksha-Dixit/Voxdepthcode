import cv2
import numpy as np
import glob
import tqdm
import os

# gtpath = "/home/yashashwee/cudaHello/dataset/MidAir/Kite_training/sunny/gt_depth/trajectory_0000/"
# gtpath = "/home/yashashwee/DeepSmooth/evals/trajectory_0001/gt/"
gtpath = "/home/yashashwee/cudaHello/dataset/Corrected/"

corpath = "/home/yashashwee/cudaHello/tempOuts/outs8/"
# corpath = "/home/yashashwee/cudaHello/dataset/LMSDepthMidair/"
# corpath = "/home/yashashwee/cudaHello/dataset/RealsenseMidair2/"

# corpath = "/home/yashashwee/DeepSmooth/evals/trajectory_0001/pred/"
# occpath = "/home/yashashwee/cudaHello/dataset/MidAir/Kite_training/sunny/stereo_occlusion/trajectory_0000/"
occpath = "/home/yashashwee/cudaHello/dataset/CorrectedOcc/"

sqList = []
for f in tqdm.tqdm(glob.glob(gtpath+"*")):
    i = f.split("/")[-1].split(".")[0]
    filledI = i.zfill(6)

    if not os.path.exists(corpath+i+".PNG"):
        continue

    # print(f)
    gtImg = cv2.imread(gtpath+ i +".tif",cv2.IMREAD_UNCHANGED) #[0].astype(np.uint16)
    occImg = cv2.imread(occpath+i+".tif",cv2.IMREAD_UNCHANGED)
    corimg = cv2.imread(corpath+i+".PNG",cv2.IMREAD_UNCHANGED) #[0].astype(np.uint16)
    # corimg = cv2.medianBlur(corimg,5)
    # print(corimg,f)
    # corimg =  cv2.normalize(corimg,None, 0, np.iinfo(np.uint16).max, cv2.NORM_MINMAX)
    # gtImg =  cv2.normalize(gtImg,None, 0, np.iinfo(np.uint16).max, cv2.NORM_MINMAX)

    # corimg = cv2.resize(corimg,dsize=(1024,1024))
    # gtImg = cv2.resize(gtImg,dsize=(1024,1024))
    if len(corimg[occImg==0])==0:
        continue
    oneArr = np.ones_like(occImg)
    # print(corimg[occImg==0]+oneArr[occImg==0],gtImg[occImg==0])
    # print(np.log(corimg[occImg==0]),np.log(gtImg[occImg==0]))
    rmse = np.mean(np.square(corimg[occImg==0]-gtImg[occImg==0]))
    sqList.append(rmse)


print("RMSE : ",np.mean(np.sqrt(sqList)))
