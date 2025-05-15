#include "opencv2/opencv.hpp"
void toSurfaceCaller(float* inputMat,float* pointCloud,int imgW,int imgH);
void toImage(float *pointCloud,short *outImg,long long sz,int imgH,int imgW);
void setDeviceMem(int imgW,int imgH);
void destroyMem();
