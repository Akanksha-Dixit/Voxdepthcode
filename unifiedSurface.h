#include <iostream>
#include<chrono>
#include <cmath>
#include "opencv2/opencv.hpp"
void toSurfaceCallerUnified();
void allocUnifiedMem(int imgW,int imgH);
void setUnifiedMem(float* inputMap);
void copyPC(float *pc);
void copyDepthImg(short* depthImg);
void destroyUnifiedMem();
void setTrans(cv::Mat &Rt);
void transformPCDCaller();
