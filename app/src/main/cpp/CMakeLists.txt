cmake_minimum_required(VERSION 3.22.1)

project("facedetect")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti")

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.5.4-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20240410-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(
        native-lib
        SHARED
        # List C/C++ source files with relative paths to this CMakeLists.txt.
        native-lib.cpp
        yolo11_pose_ncnn.cpp
        ndkcamera.cpp
        yolo11_pose.cpp
        )

find_library(
        log-lib

        log )

target_link_libraries(
        native-lib
        ${OpenCV_LIBS}
        ncnn
        jnigraphics
        camera2ndk
        mediandk
        opencv_video
        android
        log)