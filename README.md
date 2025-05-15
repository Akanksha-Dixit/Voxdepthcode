# VoxDepthCode
Code for VoxDepth, a depth completion method for embedded systems.
## Requirements
Hardware requirements :
- Jetson Nano Board
- 10W power supply

Software Requirements :
- Ubuntu >=20.04.4
- CUDA Toolkit 10.x
- OpenCV 4.x
- CMAKE >=3.18
- Python 3.x (For plots)
## Dataset
We will make our own dataset available in due time but until then feel free to test the implementation on the [MidAir](https://midair.ulg.ac.be/download.html) dataset. Remeber to resize the images down to 640x480, to get the correct latencies. You can also use our DatasetGen.py script to create your own dataset.

## Configuration 
The configs/config.txt file contains the parameters required for the processing pipeline. Each parameter can be customized to suit your dataset and processing requirements. Follow the instructions below to understand how to configure the file.

**depthPath:** Path to the directory containing noisy depth images.  
**colorPath:** Path to the directory containing color images.  
**outPath:** Directory where the processed outputs will be saved. Make sure the path exists.  
**initFrame:** Starting frame name  
**initRead:** Starting index for reading frames.  
**initWindow:** Fusion window size  
**final:** #Frames after which the template image needs to be regenerated.  
**finishOffset:** Last frame to process. Your dataset atleast should have these many images.  
**filterType:** Type of filtering applied to the depth image. Options include:  ERODE and DILATE.  
**depthCutoff:** Minimum depth value for filtering. Values below this will be ignored.  
**resizeWH:** Target width and height for resizing images.  
**isResize:** Flag to enable or disable resizing.  
**minMatch:** Minimum number of matches required to proceed with processing.  
**fastMatchThres:** Threshold for fast feature matching.  
**goodMatchThres:** Threshold for identifying good matches.  


## Build
Building VoxDepth is similar to building most CMAKE projects. Go into the VoxDepth directory and follow the following steps:

```
mkdir build
cd build
cmake ..
make
```

Make the CUDA compiler is available on your path.

## Test
You may test the code by modifying the config.txt file available in the configs folder. Please remove the print statements before testing for latencies.

```
make
./voxel
```
