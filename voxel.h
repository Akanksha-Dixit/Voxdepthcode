#include <iostream>
#include<chrono>
#include <cmath>
#include <fstream>
#include <bits/stdc++.h>
#include "opencv2/opencv.hpp"


enum FilterType {
    MEDIAN,
    ERODE,
    DILATE,
    BILINEAR,
    NONE
};



void allocUnifiedMem(int imgW,int imgH);
void resetImgSize(int imgW,int imgH);
void setUnifiedMem(float* inputMap);
void toSurfaceCallerUnified();
void copyDepthImg(short* depthImg,FilterType T);
void destroyUnifiedMem();
void surfaceToImgCaller(short* depthImg);
void setImageBuffs(float* bgd,float* fgd);
void combineDepth(short* depthImg,float depthCutoff);

