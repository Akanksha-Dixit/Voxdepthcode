#include "voxel.h"

//realsense
#define CX 322.5f
#define CY 181.4f
#define FXY 325.5f


//Midair
// #define CX 512.0f
// #define CY 512.0f
// #define FXY 512.0f

#define XYDIV 0.1f
#define ZDIV 0.065f
#define KERSIZE 5
#define USMAX 65535
#define VOXSIZE 1000
#define XYVOXMAX 100
#define ZVOXMAX 65
#define BUILDING true


int szMat;
dim3 block1,grid1;
dim3 blockVox,gridVox;
float* inputMat,* pointCloud,*transform,*outputMat;
char* voxelGrid;
int imgHit,imgWid;

////////////////////////////////////////////////////////
//
// KERNELS
//
////////////////////////////////////////////////////////
/*
__device__ static inline char atomicAdd(char* address, char val) {
    // offset, in bytes, of the char* address within the 32-bit address of the space that overlaps it
    size_t long_address_modulo = (size_t) address & 3;
    // the 32-bit address that overlaps the same memory
    auto* base_address = (unsigned int*) ((char*) address - long_address_modulo);
    // A 0x3210 selector in __byte_perm will simply select all four bytes in the first argument in the same order.
    // The "4" signifies the position where the first byte of the second argument will end up in the output.
    unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
    // for selecting bytes within a 32-bit chunk that correspond to the char* address (relative to base_address)
    unsigned int selector = selectors[long_address_modulo];
    unsigned int long_old, long_assumed, long_val, replacement;

    long_old = *base_address;

    do {
        long_assumed = long_old;
        // replace bits in long_old that pertain to the char address with those from val
        long_val = __byte_perm(long_old, 0, long_address_modulo) + val;
        replacement = __byte_perm(long_old, long_val, selector);
        long_old = atomicCAS(base_address, long_assumed, replacement);
    } while (long_old != long_assumed);
    return __byte_perm(long_old, 0, long_address_modulo);
}
*/
__global__ void kernelToSurfaceUnified(float* inputMat,char* voxelGrid,float* outputMat,int imgW,int imgH)
{
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x; //image x index or column number
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y; //image y index of row number
    if(xIndex<imgH && yIndex<imgW)
    {
        float z = inputMat[xIndex*imgW + yIndex];
        if(z!=0.0)
        {
            int xVox = (int)floor(((((float)xIndex-CX)*z)/FXY)/XYDIV) + 500;
            int yVox = (int)floor(((((float)yIndex-CY)*z)/FXY)/XYDIV) + 500;
            int zVox = (int)floor(z/ZDIV);
            
            if(xVox>=0 && yVox>=0 && zVox>=0 && xVox<VOXSIZE && yVox<VOXSIZE && zVox<VOXSIZE)
            {
                voxelGrid[xVox + yVox*VOXSIZE + zVox*VOXSIZE*VOXSIZE] = 1;
                if(BUILDING)
                {
                    float xCoor = ((float)(xVox-500)*XYDIV);
                    float yCoor = ((float)(yVox-500)*XYDIV);
                    int u = round(((xCoor)*FXY)/z + CX);
                    int v = round(((yCoor)*FXY)/z + CY);
                    float zNew = ((float)zVox)*ZDIV*1000;
                    if(u<imgH && v<imgW)
                    {
                        outputMat[u*imgW+v] = zNew;
                    }
                }
            }

        }
        
    }
}
__global__ void combineDepthKernel(float* inputMat,float* outputMat,int imgW,int imgH,float depthCutoff)
{
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x; //image x index or column number
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y; //image y index of row number
    if(xIndex<imgH && yIndex<imgW)
    {
        
        if(inputMat[xIndex*imgW + yIndex]>=65.0f || inputMat[xIndex*imgW + yIndex]==0.0f)
        {
            inputMat[xIndex*imgW + yIndex] = outputMat[xIndex*imgW + yIndex];
        }
        if(inputMat[xIndex*imgW + yIndex]<=depthCutoff)
        {
            inputMat[xIndex*imgW + yIndex] = (float) USMAX;
        }

    }
}
__global__ void kernelSurfaceToImage(char* voxelGrid,float* outputMat,int imgW,int imgH)
{
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int zIndex = blockIdx.z * blockDim.z + threadIdx.z;
    if(xIndex<1000 && yIndex<1000 && zIndex<1000)
    {
        //440,584,95
        
           
        float z = ((float)zIndex*ZDIV)*1000;
        // printf("%f\n",z);
        float xCoor = ((float)(xIndex-500)*XYDIV);
        float yCoor = ((float)(yIndex-500)*XYDIV);
        int u = round(((xCoor)*FXY)/z + CX);
        int v = round(((yCoor)*FXY)/z + CY);
        if(u<imgH && v<imgW )
        {
            if(outputMat[u*imgW+v]>z)
                outputMat[u*imgW+v] =z;
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

__global__ void bilinearInterpolation(float* input,float *output, int input_width, int input_height, int output_width, int output_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= output_width || y >= output_height) return;

    // Compute the coordinates in the input image
    float x_ratio = (float)(input_width - 1) / output_width;
    float y_ratio = (float)(input_height - 1) / output_height;
    float x_input = x * x_ratio;
    float y_input = y * y_ratio;

    int x_low = floor(x_input);
    int y_low = floor(y_input);
    int x_high = min(x_low + 1, input_width - 1);
    int y_high = min(y_low + 1, input_height - 1);

    // Compute the weights for interpolation
    float x_weight = x_input - x_low;
    float y_weight = y_input - y_low;

    // Get the pixel values at the four nearest neighbors
    float top_left = input[y_low * input_width + x_low];
    float top_right = input[y_low * input_width + x_high];
    float bottom_left = input[y_high * input_width + x_low];
    float bottom_right = input[y_high * input_width + x_high];

    // Perform bilinear interpolation
    float top = top_left + (top_right - top_left) * x_weight;
    float bottom = bottom_left + (bottom_right - bottom_left) * x_weight;
    float pixel_value = top + (bottom - top) * y_weight;

    // Write the interpolated value to the output image
    output[y * output_width + x] = pixel_value;
}

__global__ void erosionFilter(float* inputMat,float *outputMat,int imgW,int imgH,int kerSize,int replaceIdx)
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
        outputMat[xIndex*imgW + yIndex] = sorted[replaceIdx];

    }
}

//////////////////////////////////////////
//
//Caller functions
//
/////////////////////////////////////////
void allocUnifiedMem(int imgW,int imgH)
{
    imgHit = imgH;
    imgWid = imgW;
    szMat = imgH*imgW;
    block1.x = 32;
    block1.y = 32;
    grid1.x = (imgH + block1.x-1)/block1.x;
    grid1.y = (imgW + block1.y-1)/block1.y;

    blockVox.x = 10;
    blockVox.y = 10;
    blockVox.z = 10;
    gridVox.x = (1000 + blockVox.x-1)/blockVox.x;
    gridVox.y = (1000 + blockVox.y-1)/blockVox.y;
    gridVox.z = (1000 + blockVox.z-1)/blockVox.z;



    cudaMallocManaged(&inputMat,szMat*sizeof(float));
    cudaMallocManaged(&outputMat,szMat*sizeof(float));
    cudaMallocManaged(&voxelGrid,VOXSIZE*VOXSIZE*VOXSIZE*sizeof(char));
    cudaMallocManaged(&transform,16*sizeof(float));
}




void setUnifiedMem(float* inputMap)
{
    std::memcpy(inputMat,inputMap,szMat*sizeof(float));
}
void setImageBuffs(float* bgd,float* fgd)
{
    std::memcpy(outputMat,bgd,szMat*sizeof(float));
    std::memcpy(inputMat,fgd,szMat*sizeof(float));
}
void destroyUnifiedMem()
{
    cudaFree(inputMat);
    cudaFree(outputMat);
    cudaFree(voxelGrid);
    cudaFree(transform);
}

void displaySystemInfo()
{
    struct cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, 0);
    std::cout<<"using "<<properties.multiProcessorCount<<" multiprocessors"<<std::endl;
    std::cout<<"max threads per processor: "<<properties.maxThreadsPerMultiProcessor<<std::endl;
}
void toSurfaceCallerUnified()
{   
    // cudaError_t err = cudaSuccess;
    kernelToSurfaceUnified<<<grid1,block1>>>(inputMat,voxelGrid,outputMat,imgWid,imgHit);
    
    cudaDeviceSynchronize();

}

void surfaceToImgCaller(short* depthImg)
{
    for(int xIndex=0;xIndex<VOXSIZE;xIndex++)
    {
        for(int yIndex=0;yIndex<VOXSIZE;yIndex++)
        {
            for(int zIndex=0;zIndex<VOXSIZE;zIndex++)
            {   
                
                if(voxelGrid[xIndex + yIndex*VOXSIZE + zIndex*VOXSIZE*VOXSIZE] ==(char)1)
                {
                    float z = ((float)zIndex*ZDIV)*1000;
                    
                    float xCoor = ((float)(xIndex-500)*XYDIV);
                    float yCoor = ((float)(yIndex-500)*XYDIV);
                    int u = round(((xCoor)*FXY)/z + CX);
                    int v = round(((yCoor)*FXY)/z + CY);
                    if(u<imgHit && v<imgWid )
                    {
                        // printf("%f and %f\n",z,outputMat[u*imgWid+v]);
                        if(outputMat[u*imgWid+v]==0.0 || outputMat[u*imgWid+v]>z)
                        {
                            outputMat[u*imgWid+v] =z;
                            depthImg[u*imgWid+v] = std::floor(outputMat[u*imgWid+v]);

                        }
                    }
                }
            }
        }
    }
    cudaDeviceSynchronize();
}
void copyDepthImg(short* depthImg,FilterType T)
{
    if(T == ERODE)
    {
        erosionFilter<<<grid1,block1>>>(outputMat,inputMat,imgWid,imgHit,KERSIZE,KERSIZE*KERSIZE-1);
        // erosionFilter<<<grid1,block1>>>(inputMat,outputMat,imgWid,imgHit,KERSIZE,1);
        cudaDeviceSynchronize();
        for(int i=0;i<szMat;i++)
        {
            
            depthImg[i] = std::floor(inputMat[i]);
            if(depthImg[i]==0) depthImg[i] = USMAX;
            
        }
    }
    else if(T == DILATE)
    {
        erosionFilter<<<grid1,block1>>>(outputMat,inputMat,imgWid,imgHit,KERSIZE,1);
        cudaDeviceSynchronize();
        for(int i=0;i<szMat;i++)
        {
            
            depthImg[i] = std::floor(inputMat[i]);
            
        }
    }
    else if(T == BILINEAR)
    {
        bilinearInterpolation<<<grid1,block1>>>(outputMat,inputMat,imgWid,imgHit,imgWid,imgHit);
        cudaDeviceSynchronize();
        for(int i=0;i<szMat;i++)
        {
            
            depthImg[i] = std::floor(inputMat[i]);
            
        }
    }
    else
    {
        // medFilter<<<grid1,block1>>>(outputMat,inputMat,imgWid,imgHit,KERSIZE);
        cudaDeviceSynchronize();
        for(int i=0;i<szMat;i++)
        {
            
            depthImg[i] = std::floor(outputMat[i]);
            
        }
    }
    
}

void combineDepth(short* depthImg,float depthCutoff)
{
    combineDepthKernel<<<grid1,block1>>>(inputMat,outputMat,imgWid,imgHit,depthCutoff);
    medFilter<<<grid1,block1>>>(inputMat,outputMat,imgWid,imgHit,KERSIZE);
    cudaDeviceSynchronize();
    for(int i=0;i<szMat;i++)
    {
        
        depthImg[i] = std::floor(outputMat[i]*1000.0f);
        
    }
}
void resetImgSize(int imgW,int imgH)
{
    imgHit = imgH;
    imgWid = imgW;
    szMat = imgH*imgW;
    block1.x = 32;
    block1.y = 32;
    grid1.x = (imgH + block1.x-1)/block1.x;
    grid1.y = (imgW + block1.y-1)/block1.y;
}