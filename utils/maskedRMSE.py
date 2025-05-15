import cv2
import numpy as np
import glob
import tqdm

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

sqList = []
for f in glob.glob(imgsPath+"*"):
    i = f.split("/")[-1].split(".")[0]
    filledI = i.zfill(6)
    gtImg = cv2.imread(gtpath+ filledI +".PNG",cv2.IMREAD_UNCHANGED) #[0].astype(np.uint16)
    occImg = cv2.imread(occpath+filledI+".PNG",cv2.IMREAD_UNCHANGED)
    corimg = cv2.imread(corpath+filledI+".PNG",cv2.IMREAD_UNCHANGED)
    # corimg = cv2.resize(corimg,(1024,1024),None)

    # print(gtImg.shape,corimg.shape)
    # print(np.max(gtImg),np.max(corimg))
    # corimg = cv2.medianBlur(corimg,5)
    # print(corimg,f)
    # corimg =  cv2.normalize(corimg,None, 0, np.iinfo(np.uint16).max, cv2.NORM_MINMAX)
    # gtImg =  cv2.normalize(gtImg,None, 0, np.iinfo(np.uint16).max, cv2.NORM_MINMAX)

    # corimg = cv2.resize(corimg,dsize=(1024,1024))
    # gtImg = cv2.resize(gtImg,dsize=(1024,1024))
    if len(corimg[occImg==0])==0:
        continue
    rmse = np.mean(np.square(corimg[occImg==0]-gtImg[occImg==0]))
    print(f,rmse)
    sqList.append(rmse)
    # if len(sqList)>300:
    #     break


print("RMSE : ",np.mean(np.sqrt(sqList)))
