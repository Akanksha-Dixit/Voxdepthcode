cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")


project(hello LANGUAGES CXX CUDA)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_library(myHelper STATIC helpers.cpp)
target_include_directories(myHelper PUBLIC ${PROJECT_SOURCE_DIR})


add_executable(test main.cpp toSurface.cu)
target_link_libraries(test PRIVATE cudart ${OpenCV_LIBS})

add_executable(unified testUnified.cpp unifiedSurface.cu)
target_link_libraries(unified PRIVATE cudart ${OpenCV_LIBS} myHelper)

add_executable(voxel voxel.cpp voxel.cu)
target_link_libraries(voxel PRIVATE cudart ${OpenCV_LIBS} myHelper)
