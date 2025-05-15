#include<iostream>
#include "opencv2/opencv.hpp"
#include <opencv2/rgbd/depth.hpp>
#include <opencv2/features2d.hpp>
#include "toSurface.h"
#include<chrono>



using namespace std;
using namespace cv;
// const float vals[] = 

// Ptr<rgbd::RgbdOdometry> odo = rgbd::RgbdOdometry::create();

int main()
{
    Mat img,colorImage;

    string pth = "/home/yashashwee/DepthFill/dataset/MidAir/Kite_training/sunny/occluded_depth/train/";
    string grayPth = "/home/yashashwee/DepthFill/dataset/MidAir/Kite_training/sunny/color_left/train/";

    
    Mat_<float> flImg,flColor;
    float *matDepth,*pointCloud;

    img = imread(pth+"002203.PNG",IMREAD_UNCHANGED);
    flImg=  img/1000.0f;
    colorImage = imread(grayPth+"002203.JPEG",IMREAD_UNCHANGED);
    flColor = colorImage/255.0f;

    
    
    matDepth = (float *)flImg.data;
    int imgW = flImg.cols, imgH = flImg.rows;
    int sz = imgH*imgW;
    pointCloud = new float[3*(sz)];
    

    // cout<<flImg.at<float>(10,21)<<" "<<matDepth[10*flImg.cols+21]<<"\n";
    setDeviceMem(flImg.cols,flImg.rows);
    auto start = chrono::high_resolution_clock::now();

    toSurfaceCaller(matDepth,pointCloud,flImg.cols,flImg.rows);
    
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    cout<<duration.count()<<endl;
    destroyMem();

    short  *outPtr = new short  [imgH*imgW];
    for(int i=0;i<imgH*imgW;i++)
        outPtr[i]=65535;
    toImage(pointCloud,outPtr,sz,imgH,imgW);
    
    Mat imgOut;
    img.copyTo(imgOut);
    for(int i=0;i<imgH;i++)
    {
        for(int j=0;j<imgW;j++)
        {
            imgOut.at<short>(i,j) = (short) outPtr[(i*imgW+j)];
        }
    }
    

    imwrite("../outs/out2.PNG",imgOut);
    return 0;
}