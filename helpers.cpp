#include "helpers.h"

void zfill(std::string &str,int fill)
{
    int len = str.length();
    for(int i=0;i<fill-len;i++)
        str += str;
    
}
void readImg(std::string &path,int id,cv::Mat &img,cv::Mat_<float> &flImg,float div,std::string ext)
{
    std::string strId = std::to_string(id);
    zfill(strId,6);
    std::cout<<path+strId+"."+ext<<std::endl;
    // img = cv::imread(path+std::to_string())
}