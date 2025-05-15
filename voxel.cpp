#include<iostream>
#include "opencv2/opencv.hpp"
#include <opencv2/rgbd/depth.hpp>
#include<chrono>
#include<thread>
#include "voxel.h"

using namespace std;
using namespace cv;

// Map to convert filter type strings to enum values
std::map<std::string,FilterType> enumMap = {
    {"MEDIAN",MEDIAN},
    {"ERODE",ERODE},
    {"DILATE",DILATE},
    {"BILINEAR",BILINEAR},
    {"NONE",NONE}
};

// Function to read configuration file into a map
std::map<std::string, std::string> readConfig(const std::string& filename) {
    std::map<std::string, std::string> config;
    std::ifstream file(filename);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            size_t delimiterPos = line.find('=');
            if (delimiterPos != std::string::npos) {
                std::string key = line.substr(0, delimiterPos);
                std::string value = line.substr(delimiterPos + 1);
                config[key] = value;
            }
        }
        file.close();
    }
    return config;
}


float vals[] = {325.436f,0.0f,322.527f,0.0f,325.436f,181.355f,0.0f,0.0f,1.0f};
const Mat cameraMat = Mat(3,3,CV_32FC1,vals);

//RGB-D and visual registration params
int iterCnt[] = {10,0,0,0}; // Iteration counts for different steps
Ptr<FastFeatureDetector> fast = FastFeatureDetector::create(100,false); // FAST feature detector
Ptr<BFMatcher> bf = BFMatcher::create(NORM_HAMMING); // Brute-force matcher
Ptr<ORB> orb = ORB::create(); // ORB feature detector
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

// Function declarations
void setEmpty(short *sptr,int imgW,int imgH);
void zfill(std::string &str,int fill);
void readImg(std::string &path,int id,cv::Mat &img,cv::Mat_<float> &flImg,float div,std::string ext,bool isColor,bool resizeImg,int resizeWH);
void setFtrMap(Mat &mp,int imgH,int imgW);

int main()
{
    //Read config
    
    map<std::string,std::string> confMap = readConfig("./configs/configs.txt"); // Replace this with the exact full path of the config.txt file
    cout<<"Printing Params\n";

    // Print configuration parameters
    for(auto m:confMap)
    {
        std::cout<<m.first<<" "<<m.second<<std::endl;
    }
    cout<<"\n\n";
    
    // Update FAST and ORB parameters based on configuration
    fast->setThreshold(stoi(confMap["fastMatchThres"]));
    orb->setFastThreshold(stoi(confMap["fastMatchThres"]));
    orb->setNLevels(3);

    // Resize images according to the configuration settings
    int resz = std::stoi(confMap["resizeWH"]);
    int isResz = confMap["isResize"]=="1";
    cv::Size cvResz(resz,resz);
    
    // Initialize RGB-D odometry
    odo->setIterationCounts(iterCntMat);

    // Setting paths for input images
    Mat img,colorImage;
    Mat_<float> flImg,flImgBase,flColor;
    string pth = confMap["depthPath"];
    string grayPth = confMap["colorPath"];


    // float *matDepth,*pointCloud;

    // Reading input images
    img = imread(pth+confMap["initFrame"]+".PNG",IMREAD_UNCHANGED);
    flImg=  img/1000.0f;  // Normalize depth
    colorImage = imread(grayPth+confMap["initFrame"]+".JPEG",IMREAD_GRAYSCALE);
    flColor = colorImage/255.0f; // Normalize color

    int imgW = flImg.cols, imgH = flImg.rows;
    cv::Size ogSz(imgW,imgH);
    int minMatch = stoi(confMap["minMatch"]);
    int goodMatchThres = stoi(confMap["goodMatchThres"]);
    short  *outPtr = new short  [imgH*imgW];
    
    // Creating feature maps
    Mat curFtrMap,futFtrMap;
    curFtrMap = Mat::zeros(flImg.size(),CV_8U);
    futFtrMap = Mat::zeros(flImg.size(),CV_8U);
    setFtrMap(curFtrMap,imgH,imgW);
    setFtrMap(futFtrMap,imgH,imgW);
    // }
    
    int beg =std::stoi(confMap["initRead"]);

    // Reading fusion window size
    const int window=std::stoi(confMap["initWindow"]),final=std::stoi(confMap["final"]),finishIdx = beg+std::stoi(confMap["finishOffset"]); 
    
    // Main processing loop
    auto startTot = chrono::high_resolution_clock::now();
    initPhase:
    img = imread(pth+confMap["initFrame"]+".PNG",IMREAD_UNCHANGED);
    flImg=  img/1000.0f;
    colorImage = imread(grayPth+confMap["initFrame"]+".JPEG",IMREAD_GRAYSCALE);
    flColor = colorImage/255.0f;
    setEmpty(outPtr,imgW,imgH);

    // Unified memory setup (CUDA implementation)
    allocUnifiedMem(imgW,imgH);
    
    auto start = chrono::high_resolution_clock::now();
    // Calling CUDA function
    toSurfaceCallerUnified();
    
    //Step 1: Point Cloud Fusion
    //Takes the first n (fusion windw size) frames to create a fused point cloud that represents an accurate 3D representation of the scene.
    for(int i=beg;i<=beg+window;i++)
    {   
        // Process each frame in the fusion window
        Mat colorImageD;
        Mat_<float> flImgD,flColorD;
        readImg(pth,i,img,flImgD,1000.0f,"PNG",false,false,resz);
        readImg(grayPth,i,colorImageD,flColorD,255.0f,"JPEG",true,false,resz);
        // std::cout<<colorImage.size()<<":"<<flImg.size()<<"\n";
        
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

    //copy and upsample
    auto upsampleStart = chrono::high_resolution_clock::now();
    copyDepthImg(outPtr,enumMap[confMap["filterType"]]);
    auto upsampleEnd = chrono::high_resolution_clock::now();
    auto durationSample = chrono::duration_cast<chrono::milliseconds>(upsampleEnd-upsampleStart);
    std::cout<<"Time for upsample "<<durationSample.count()<<" milliseconds"<<endl;


    //Stop the depth correction process when finishOffset is reached
    if(beg >= finishIdx)
    {
        destroyUnifiedMem();
        auto stopTot = chrono::high_resolution_clock::now();
        auto durationTot = chrono::duration_cast<chrono::milliseconds>(stopTot - startTot);
        std::cout<<"Total time taken: "<<durationTot.count()<<" milliseconds"<<endl;
        return 0;
    }

    Mat imgOut = Mat(img.rows,img.cols,CV_16UC1,(unsigned*)outPtr);
    cv::imwrite("../voxelOut/out"+to_string(beg)+".PNG",imgOut);

    if(isResz)
    {
        cv::resize(imgOut,imgOut,cvResz);
        cv::resize(colorImage,colorImage,cvResz);
    }


    //Step 2: Depth correction

    vector<KeyPoint> keypoints1,keypoints2;
    Mat descriptors1,descriptors2;
    fast->clear();
    orb->clear();
    bf->clear();

    //Extract features and keypoints of the template image
    fast->detect(colorImage,keypoints1);
    orb->compute(colorImage,keypoints1,descriptors1);

    Mat affineTrans;
    int minGood = INT16_MAX;
    std::cout<<beg+window+1<<" start combining\n";

    //Multi-threading to accelerate the depth correction process
    thread th1(readImg,std::ref(pth),beg+window+1,std::ref(img),std::ref(flImg),1000.0f,"PNG",false,isResz,resz);
    thread th2(readImg,std::ref(grayPth),beg+window+1,std::ref(colorImage),std::ref(flColor),255.0f,"JPEG",true,isResz,resz);
    
    if(isResz)
        resetImgSize(resz,resz);
    th1.join();
    th2.join();
    
    for(int i=beg+window+1;i<beg+window+final;i++)
    {
        
            vector<DMatch> matches;
            vector<Point2f> points1,points2;
            
            
            fast->clear();
            orb->clear();
            bf->clear();
            auto startPart = chrono::high_resolution_clock::now();

            //Extract features and keypoints of the noisy image
            fast->detect(colorImage,keypoints2);
            orb->compute(colorImage,keypoints2,descriptors2);

            //Match features of the template image and the noisy image and find #good matches
            bf->match(descriptors1,descriptors2,matches);
            int goodMatches = 0;
            
            for(const auto& match : matches)
            {   
                if(match.distance<=goodMatchThres)
                {
                    goodMatches++;
                    // continue;
                }
                points1.push_back(keypoints1[match.queryIdx].pt);
                points2.push_back(keypoints2[match.trainIdx].pt);
                
            }
            minGood = minGood > goodMatches ? goodMatches:minGood;
            // std::cout<<goodMatches<<std::endl;

            //If #good matches < minmatch, recompute the template image
            if(goodMatches<minMatch)
            {
                std::cout<<"Breaking at "<<i<<" min good matches "<<minGood<<"\n";
                beg = i-window;
                destroyUnifiedMem();
                //recompute the template image
                goto initPhase;  

            }
            affineTrans = estimateAffine2D(points1, points2);
        

        thread th1(readImg,std::ref(pth),i+1,std::ref(img),std::ref(flImg),1000.0f,"PNG",false,isResz,resz);
        thread th2(readImg,std::ref(grayPth),i+1,std::ref(colorImage),std::ref(flColor),255.0f,"JPEG",true,isResz,resz);
        
        
        cv::warpAffine(imgOut,imgOut,affineTrans,imgOut.size());
        
        keypoints1 = keypoints2;
        descriptors1 = descriptors2;
        

        imwrite("../outsBGM/"+to_string(i)+".PNG",imgOut);

        
        flImgBase = imgOut/1000.0f;
        setImageBuffs((float *)flImgBase.data,(float *)flImg.data);
        //Combine the depth to produce the corrected depth image
        combineDepth(outPtr,stof(confMap["depthCutoff"]));
        
        
        Mat imgCombine = Mat(img.rows,img.cols,CV_16UC1,(unsigned*)outPtr);
        if(isResz)
            cv::resize(imgCombine,imgCombine,ogSz,0.0,0.0,cv::INTER_CUBIC);
        
        imwrite(confMap["outPath"]+to_string(i)+".PNG",imgCombine);
        th1.join();
        th2.join();
        

    }
    std::cout<<"Breaking at "<<beg+window+final-1<<" min good matches "<<minGood<<"\n";
    beg = beg+window+final;
    destroyUnifiedMem();
    goto initPhase;
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
void readImg(std::string &path,int id,cv::Mat &img,cv::Mat_<float> &flImg,float div,std::string ext,bool isColor,bool resizeImg,int resizeWH)
{
    std::string strId = std::to_string(id);
    zfill(strId,6);
    // std::cout<<path+strId+"."+ext<<std::endl;
    // cout<<"ReadImg\n";
    if (isColor)    
        img = cv::imread(path+strId+"."+ext,cv::IMREAD_GRAYSCALE);
    else
        img = cv::imread(path+strId+"."+ext,cv::IMREAD_UNCHANGED);
    if(resizeImg)
    {
        cv::Size resz(resizeWH,resizeWH);
        cv::resize(img,img,resz);
    }
    flImg = img/div;
}

void setEmpty(short *sptr,int imgW,int imgH)
{
    for(int i=0;i<imgH*imgW;i++)
        sptr[i]=65535;
    
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