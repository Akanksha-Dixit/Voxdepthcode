#include "toSurface.h"
#include <iostream>
#include<chrono>
#include <cmath>

#define CX 322.5f
#define CY 181.4f
#define FXY 325.5f

__global__ void kernelToSurface(float* inputMat,float *pointCloud,int imgW,int imgH)
{
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x; //image x index or column number
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y; //image y index of row number
    if(xIndex<imgH && yIndex<imgW)
    {
        float z = inputMat[xIndex*imgW + yIndex];
        if(z==0.0f)
        {
            pointCloud[3*(xIndex*imgW + yIndex)] = 0.0f;
            pointCloud[3*(xIndex*imgW + yIndex) + 1] = 0.0f;
            pointCloud[3*(xIndex*imgW + yIndex) + 2] = 0.0f;
        }
        else
        {
            pointCloud[3*(xIndex*imgW + yIndex)] = (((float)xIndex-CX)*z)/FXY;
            pointCloud[3*(xIndex*imgW + yIndex) + 1] = (((float)yIndex-CY)*z)/FXY;
            pointCloud[3*(xIndex*imgW + yIndex) + 2] = z;
        }
        // printf("(%d,%d):(%f,%f,%f)\n",xIndex,yIndex,pointCloud[3*(xIndex*imgW + yIndex)],pointCloud[3*(xIndex*imgW + yIndex)+1],z);
    }
}

__global__ void testKernel(float* inputMat,float* pointCloud,int imgW,int imgH)
{
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x; //image x index or column number
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y; //image y index of row number
    if(xIndex<imgH && yIndex<imgW)
    {
        inputMat[xIndex*imgW + yIndex] = 0.0f;
        // printf("%f",inputMat[xIndex*imgW + yIndex]);
        pointCloud[3*(xIndex*imgW + yIndex)] = 11.0f;
        pointCloud[3*(xIndex*imgW + yIndex) + 1] = 11.0f;
        pointCloud[3*(xIndex*imgW + yIndex) + 2] = 11.0f;
    }
}

void toImage(float *pointCloud,short *outImg,long long sz,int imgH,int imgW)
{
    for(int i=0;i<sz;i++)
    {
        float z = pointCloud[3*i+2];
        float x = pointCloud[3*i];
        float y = pointCloud[3*i+1];
        if(z==0.0f)
            continue;
        int d = std::floor(z*1000);
        int u = std::round((x*FXY)/z + CX);
        int v = std::round((y*FXY)/z + CY);
        if(u<imgH and v<imgW)
                outImg[u*imgW + v]=d;
    }

}

int sz;
float *d_inputMat,*d_pointCloud;
dim3 block,grid;

void setDeviceMem(int imgW,int imgH)
{
    std::cout<<"Memset\n";
    sz = imgH*imgW;
    cudaMalloc(&d_inputMat,sz*sizeof(float));
    cudaMalloc(&d_pointCloud,3*(sz)*sizeof(float));
    block.x = 32;
    block.y = 32;
    grid.x = (imgH + block.x-1)/block.x;
    grid.y = (imgW + block.y-1)/block.y;
}

void toSurfaceCaller(float* inputMat,float* pointCloud,int imgW,int imgH)
{
    cudaMemcpy(d_inputMat, inputMat, sz * sizeof(float), cudaMemcpyHostToDevice);
    kernelToSurface<<<grid,block>>>(d_inputMat,d_pointCloud,imgW,imgH);
    cudaMemcpy(pointCloud,d_pointCloud,3*sz*sizeof(float),cudaMemcpyDeviceToHost);
}


void destroyMem()
{
    cudaFree(d_inputMat);
    cudaFree(d_pointCloud);
}