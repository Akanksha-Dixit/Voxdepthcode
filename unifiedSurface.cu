#include "unifiedSurface.h"



#define CX 322.5f
#define CY 181.4f
#define FXY 325.5f
#define KERSIZE 5
#define USMAX 65535


int szMat;
dim3 block1,grid1;
float* inputMat,* pointCloud,*transform,*outputMat;
int imgHit,imgWid;
void allocUnifiedMem(int imgW,int imgH)
{
    imgHit = imgH;
    imgWid = imgW;
    szMat = imgH*imgW;
    block1.x = 32;
    block1.y = 32;
    grid1.x = (imgH + block1.x-1)/block1.x;
    grid1.y = (imgW + block1.y-1)/block1.y;
    
    cudaMallocManaged(&inputMat,szMat*sizeof(float));
    cudaMallocManaged(&outputMat,szMat*sizeof(float));
    cudaMallocManaged(&pointCloud,3*szMat*sizeof(float));
    cudaMallocManaged(&transform,16*sizeof(float));
}
void setUnifiedMem(float* inputMap)
{
    std::memcpy(inputMat,inputMap,szMat*sizeof(float));
}
void setTrans(cv::Mat &Rt)
{
    std::cout<<transform[0]<<" "<<transform[1]<<" "<<transform[2]<<" "<<transform[3]<<std::endl;
    std::cout<<transform[4]<<" "<<transform[5]<<" "<<transform[6]<<" "<<transform[7]<<std::endl;
    std::cout<<transform[8]<<" "<<transform[9]<<" "<<transform[10]<<" "<<transform[11]<<std::endl;
}
__global__ void transformPCD(float* pointCloud,float* transform,int imgW,int imgH)
{
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x; //image x index or column number
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y; //image y index of row number
    if(xIndex<imgH && yIndex<imgW)
    {
        
        float x = pointCloud[3*(xIndex*imgW + yIndex)];
        float y = pointCloud[3*(xIndex*imgW + yIndex) + 1];
        float z = pointCloud[3*(xIndex*imgW + yIndex) + 2];
        // printf("Before (%d,%d):(%f,%f,%f)\n",xIndex,yIndex,pointCloud[3*(xIndex*imgW + yIndex)],pointCloud[3*(xIndex*imgW + yIndex)+1],z);

        pointCloud[3*(xIndex*imgW + yIndex)] = x*transform[0] + y*transform[1] + z*transform[2] + transform[3];
        pointCloud[3*(xIndex*imgW + yIndex) + 1] = x*transform[4] + y*transform[5] + z*transform[6] + transform[7];
        pointCloud[3*(xIndex*imgW + yIndex) + 2] = x*transform[8] + y*transform[9] + z*transform[10] + transform[11];

        printf("After (%d,%d):(%f,%f,%f,%f)-(%f,%f,%f):(%f,%f,%f)\n",xIndex,yIndex,transform[0],transform[1],transform[2],transform[3],x,y,z,pointCloud[3*(xIndex*imgW + yIndex)],pointCloud[3*(xIndex*imgW + yIndex)+1],pointCloud[3*(xIndex*imgW + yIndex)+2]);
    }

}
__global__ void kernelToSurfaceUnified(float* inputMat,float *outputMat,float *pointCloud,int imgW,int imgH,float cutoff)
{
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x; //image x index or column number
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y; //image y index of row number
    if(xIndex<imgH && yIndex<imgW)
    {
        float z = inputMat[xIndex*imgW + yIndex];
        if(z<=cutoff)
        {
            pointCloud[3*(xIndex*imgW + yIndex)] = 0.0f;
            pointCloud[3*(xIndex*imgW + yIndex) + 1] = 0.0f;
            pointCloud[3*(xIndex*imgW + yIndex) + 2] = 0.0f;
            outputMat[xIndex*imgW + yIndex] = (float)USMAX;
        }
        else if(pointCloud[3*(xIndex*imgW + yIndex) + 2]>z || pointCloud[3*(xIndex*imgW + yIndex) + 2]==0.0f)
        {
            pointCloud[3*(xIndex*imgW + yIndex)] = (((float)xIndex-CX)*z)/FXY;
            pointCloud[3*(xIndex*imgW + yIndex) + 1] = (((float)yIndex-CY)*z)/FXY;
            pointCloud[3*(xIndex*imgW + yIndex) + 2] = z;
            outputMat[xIndex*imgW + yIndex] = z*1000;
            // printf("(%d,%d):(%f,%f,%f)\n",xIndex,yIndex,pointCloud[3*(xIndex*imgW + yIndex)],pointCloud[3*(xIndex*imgW + yIndex)+1],z);
        }
        
    }
}

__global__ void medFilter(float* inputMat,float *outputMat,int imgW,int imgH,int kerSize)
{
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x; //image x index or column number
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y; //image y index of row number
    int halfKer = kerSize/2;
    int cnt = kerSize*kerSize;
    if(xIndex<(imgH-halfKer-1) && yIndex<(imgW-halfKer-1) && xIndex>halfKer+1 && yIndex>halfKer+1)
    {
        float sorted[KERSIZE*KERSIZE];
        int initCnt = 0;
        //store vals
        for(int i=xIndex-halfKer;i<=xIndex+halfKer;i++)
        {
            for(int j=yIndex-halfKer;j<=yIndex+halfKer;j++)
            {
                sorted[initCnt] = inputMat[i*imgW + j];
                initCnt+=1;
            }
             
        }
        
        //sort vals
        for(int i=0;i<cnt;i++)
        {
            for(int j=0;j<cnt;j++)
            {
                if(sorted[j]>sorted[j+1])
                {
                    float temp = sorted[j];
                    sorted[j] = sorted[j+1];
                    sorted[j+1] = temp;
                }
            }
        }

        //replace value
        outputMat[xIndex*imgW + yIndex] = sorted[cnt/2];

    }
}

void destroyUnifiedMem()
{
    cudaFree(inputMat);
    cudaFree(outputMat);
    cudaFree(pointCloud);
    cudaFree(transform);
}
void toSurfaceCallerUnified()
{

    kernelToSurfaceUnified<<<grid1,block1>>>(inputMat,outputMat,pointCloud,imgWid,imgHit,5.0f);
    cudaDeviceSynchronize();
    
}
void transformPCDCaller()
{
    transformPCD<<<grid1,block1>>>(pointCloud,transform,imgWid,imgHit);
    cudaDeviceSynchronize();
}
void copyPC(float *pc)
{
    std::memcpy(pc,pointCloud,3*szMat*sizeof(float));
}
void copyDepthImg(short* depthImg)
{

    medFilter<<<grid1,block1>>>(outputMat,inputMat,imgWid,imgHit,KERSIZE);
    cudaDeviceSynchronize();

    for(int i=0;i<szMat;i++)
    {
        depthImg[i] = std::floor(outputMat[i]);
    }

    // for(int i=0;i<szMat;i++)
    // {
    //     float z = pointCloud[3*i+2];
    //     float x = pointCloud[3*i];
    //     float y = pointCloud[3*i+1];
    //     // std::cout<<z<<"\n";
    //     if(z==0.0f)
    //         continue;
    //     int d = std::floor(z*1000);
    //     int u = std::round((x*FXY)/z + CX);
    //     int v = std::round((y*FXY)/z + CY);
    //     if(u<imgHit and v<imgWid)
    //     {
    //         depthImg[u*imgWid + v]=d;
    //     }
    // }
}