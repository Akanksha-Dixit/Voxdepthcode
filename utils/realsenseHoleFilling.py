import cv2
import numpy as np
import glob
import tqdm

inpath = "/home/yashashwee/cudaHello/dataset/MidAir/PLE_training/winter/occluded_depth/trajectory_6000/"
outpath = "/home/yashashwee/cudaHello/dataset/RealsenseMidair6000/"

f = "/home/yashashwee/cudaHello/dataset/MidAir/Kite_training/sunny/occluded_depth/trajectory_0002/000001.PNG"

mxVal = np.iinfo(np.uint16).max


for f in tqdm.tqdm(glob.glob(inpath+"*")):
    idx = f.split("/")[-1]
    img = cv2.imread(f,cv2.IMREAD_UNCHANGED)
    imgNew = np.ones_like(img)
    imgNew = imgNew*mxVal
    img = cv2.medianBlur(img,5)
    # print(img.shape)
    for i in range(img.shape[0]-1):
        for j in range(img.shape[1]-1):
            imgNew[i,j] = np.max([img[i,j-1],img[i-1,j],img[i-1,j-1],img[i,j+1],img[i-1,j+1]])
            # if img[i,j]== mxVal and j>0 and i>0:
            #     imgNew[i,j] == np.min([img[i,j-1],img[i-1,j],img[i-1,j-1],img[i,j+1],img[i-1,j+1]])
            # else:
            #     imgNew[i,j]=img[i,j]

    # print(outpath+idx)
    cv2.imwrite(outpath+idx,imgNew)
        