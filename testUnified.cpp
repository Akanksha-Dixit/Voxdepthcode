#include<iostream>
#include "opencv2/opencv.hpp"
#include <opencv2/rgbd/depth.hpp>
#include "unifiedSurface.h"
#include<chrono>
#include<thread>





using namespace std;
using namespace cv;

//Camera params
float vals[] = {325.436f,0.0f,322.527f,0.0f,325.436f,181.355f,0.0f,0.0f,1.0f};
const Mat cameraMat = Mat(3,3,CV_32FC1,vals);

//RGB-D and visual registration params
int iterCnt[] = {10,0,0,0};
Ptr<FastFeatureDetector> fast = FastFeatureDetector::create(40);
Ptr<BFMatcher> bf = BFMatcher::create(NORM_HAMMING);
Ptr<ORB> orb = ORB::create();
const Mat iterCntMat = Mat(4,1,CV_8UC1,iterCnt);
Ptr<rgbd::RgbdOdometry> odo = rgbd::RgbdOdometry::create(
    cameraMat,
    rgbd::Odometry::DEFAULT_MIN_DEPTH(),
    rgbd::Odometry::DEFAULT_MAX_DEPTH(),
    rgbd::Odometry::DEFAULT_MAX_DEPTH_DIFF(),
    std::vector<int>(),
    std::vector<float>(),
    rgbd::Odometry::DEFAULT_MAX_POINTS_PART(),
    rgbd::Odometry::RIGID_BODY_MOTION
);
Mat Rt = Mat(4,4, CV_32FC1);
Mat initRt = Mat(4,4, CV_64FC1);


//Helper function for image manip
void setEmpty(short *sptr,int imgW,int imgH);
void zfill(std::string &str,int fill);
void readImg(std::string &path,int id,cv::Mat &img,cv::Mat_<float> &flImg,float div,std::string ext,bool isColor);
void setFtrMap(Mat &mp,int imgH,int imgW);
void copyToOut(Mat img,short *outPtr);

int main()
{
    auto startTot = chrono::high_resolution_clock::now();

    odo->setIterationCounts(iterCntMat);
    

    //Initialize variables
    Mat img,colorImage;
    Mat_<float> flImg,flColor;
    string pth = "/home/yashashwee/DepthFill/dataset/MidAir/Kite_training/sunny/occluded_depth/train/";
    string grayPth = "/home/yashashwee/DepthFill/dataset/MidAir/Kite_training/sunny/color_left/train/";
    float *matDepth,*pointCloud;
    img = imread(pth+"002203.PNG",IMREAD_UNCHANGED);
    flImg=  img/1000.0f;
    colorImage = imread(grayPth+"002203.JPEG",IMREAD_GRAYSCALE);
    flColor = colorImage/255.0f;
    int imgW = flImg.cols, imgH = flImg.rows;
    int sz = imgH*imgW;
    short  *outPtr = new short  [imgH*imgW];
    setEmpty(outPtr,imgW,imgH);
    allocUnifiedMem(imgW,imgH);
    Mat curFtrMap = Mat::zeros(flImg.size(),CV_8U);
    Mat futFtrMap = Mat::zeros(flImg.size(),CV_8U);
    setFtrMap(curFtrMap,imgH,imgW);
    setFtrMap(futFtrMap,imgH,imgW);

    auto start = chrono::high_resolution_clock::now();
    setUnifiedMem((float *)flImg.data);
    toSurfaceCallerUnified();
    const int beg =2204,window=9,final=100; 
    for(int i=beg;i<=beg+9;i++)
    {   
        Mat colorImageD;
        Mat_<float> flImgD,flColorD;
        readImg(pth,i,img,flImgD,1000.0f,"PNG",false);
        readImg(grayPth,i,colorImageD,flColorD,255.0f,"JPEG",true);
        
        if(i==beg)
        {
            odo->compute(
                colorImage,
                flImg,
                curFtrMap,
                colorImageD,
                flImgD,
                futFtrMap,
                Rt,
                Mat()
            );
        }
        setUnifiedMem((float *)flImgD.data);
        toSurfaceCallerUnified();
        colorImage = colorImageD;
    }

    
    

    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    std::cout<<"Convert to point cloud: "<<duration.count()<<" milliseconds"<<endl;
    

    copyDepthImg(outPtr);
    destroyUnifiedMem();
    

    Mat imgOut = Mat(img.rows,img.cols,CV_16UC1,(unsigned*)outPtr);
    imwrite("../outs/out.PNG",imgOut);

    // return 0;

    vector<KeyPoint> keypoints1,keypoints2;
    Mat descriptors1,descriptors2;
    
    fast->detect(colorImage,keypoints1);
    orb->compute(colorImage,keypoints1,descriptors1);

    Mat affineTrans;
    for(int i=beg+window+1;i<beg+final;i++)
    {
        if(true)
        {
            vector<DMatch> matches;
            vector<Point2f> points1,points2;
            readImg(pth,i,img,flImg,1000.0f,"PNG",false);
            readImg(grayPth,i,colorImage,flColor,255.0f,"JPEG",true);
            
            fast->detect(colorImage,keypoints2);
            orb->compute(colorImage,keypoints2,descriptors2);
            bf->match(descriptors1,descriptors2,matches);

            for(const auto& match : matches)
            {   
                points1.push_back(keypoints1[match.queryIdx].pt);
                points2.push_back(keypoints2[match.trainIdx].pt);
            }
            
            affineTrans = estimateAffine2D(points1, points2);
        }
        warpAffine(imgOut,imgOut,affineTrans,imgOut.size());
        // add(img,imgOut,img);
        imwrite("../outs/"+to_string(i)+".PNG",imgOut);
        keypoints1 = keypoints2;
        descriptors1 = descriptors2;

    }

    auto stopTot = chrono::high_resolution_clock::now();
    auto durationTot = chrono::duration_cast<chrono::milliseconds>(stopTot - startTot);
    std::cout<<"Total time taken: "<<durationTot.count()<<" milliseconds"<<endl;
    
    return 0;
}



void zfill(std::string &str,int fill)
{
    int len = str.length();
    for(int i=0;i<fill-len;i++)
        str = "0"+ str;
    
}
void readImg(std::string &path,int id,cv::Mat &img,cv::Mat_<float> &flImg,float div,std::string ext,bool isColor)
{
    std::string strId = std::to_string(id);
    zfill(strId,6);
    // std::cout<<path+strId+"."+ext<<std::endl;
    // cout<<"ReadImg\n";
    if (isColor)    
        img = cv::imread(path+strId+"."+ext,cv::IMREAD_GRAYSCALE);
    else
        img = cv::imread(path+strId+"."+ext,cv::IMREAD_UNCHANGED);
    flImg = img/div;
}

void setEmpty(short *sptr,int imgW,int imgH)
{
    for(int i=0;i<imgH*imgW;i++)
        sptr[i]=65535;
    
}
void copyToOut(Mat img,short *outPtr)
{
    
}
void setFtrMap(Mat &mp,int imgH,int imgW)
{
    for(int i=0;i<imgH/2;i++)
    {
        for(int j=0;j<imgW/2;j++)
        {
            Point3_<uchar>* p1 = mp.ptr<Point3_<uchar>>(j,i);
            p1->x=1;
            p1->y=1;
            p1->z=1;
        }
    }
}