# Make sure that
    # Intel Realsense camera is connected and detected.
    # pyrealsense is installed (use pip install ....)
    # Intel RealSense SDK 2.0 is installed


import pyrealsense2 as rs
import numpy as np
import cv2
import os

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configure depth and color streams
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# Start streaming
pipeline.start(config)
idx = 0
try:
    while True:
        print("Waiting for frames")
        # Wait for a coherent set of frames
        frames = pipeline.wait_for_frames()
        
        # Get depth and color frames
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue
        
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())


        # Normalize depth image for better visualization (optional)
        normalized_depth = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        # Convert the normalized depth to 8-bit
        depth_8bit = np.uint8(normalized_depth)
        # Apply a colormap (e.g., COLORMAP_JET, COLORMAP_PLASMA, etc.)
        color_mapped_depth = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
        
        
        
        # Show both images
        cv2.imshow('Depth Image', color_mapped_depth)
        cv2.imshow('RGB Image', color_image)
        
        
        # Save the frames (optional: save images only once to avoid overwriting)
        # Make sure these paths exist if not create them
        cv2.imwrite(os.path.join("depth",str(idx).zfill(6)+'.PNG'), color_mapped_depth)
        cv2.imwrite(os.path.join("demo", "occluded_depth","train", str(idx).zfill(6)+'.PNG'), depth_image)
        cv2.imwrite(os.path.join("demo","color_left","train", str(idx).zfill(6)+'.JPEG'), color_image)
        idx = idx + 1
        
        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the pipeline
    pipeline.stop()
    cv2.destroyAllWindows()
