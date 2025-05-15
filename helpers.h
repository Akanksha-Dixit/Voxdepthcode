#include <iostream>
#include<chrono>
#include <cmath>
#include "opencv2/opencv.hpp"

void readImg(std::string path,int id,cv::Mat &img,cv::Mat_<float> &flImg,float div,std::string ext);