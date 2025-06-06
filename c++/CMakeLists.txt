cmake_minimum_required(VERSION 3.10.0)
project(detect VERSION 0.1.0 LANGUAGES C CXX)

# Set CUDA path
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")

# TensorRT paths
set(TensorRT_INCLUDE_DIR "/usr/include/x86_64-linux-gnu")
set(TensorRT_LIB_DIR "/usr/lib/x86_64-linux-gnu")

# Main TensorRT libraries
set(TensorRT_LIBS
    nvinfer
    nvinfer_plugin
)

# Find OpenCV
find_package(OpenCV REQUIRED)

message(STATUS "TensorRT include dir: ${TensorRT_INCLUDE_DIR}")
message(STATUS "TensorRT lib dir: ${TensorRT_LIB_DIR}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

file(GLOB SOURCES "src/*.cpp")

add_executable(detect ${SOURCES})

# Include directories
target_include_directories(detect PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIR}
    ${CUDA_TOOLKIT_ROOT_DIR}/include
)

# Link libraries
foreach(lib ${TensorRT_LIBS})
    find_library(${lib}_LIBRARY NAMES ${lib} PATHS ${TensorRT_LIB_DIR})
    list(APPEND TensorRT_LINK_LIBS ${${lib}_LIBRARY})
endforeach()

target_link_libraries(detect PRIVATE
    ${OpenCV_LIBS}
    ${TensorRT_LINK_LIBS}
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so
    pthread
    dl
)

# Add CUDA and TensorRT definitions
target_compile_definitions(detect PRIVATE
    -DNV_TENSORRT_MAJOR=10
    -DNV_TENSORRT_MINOR=9
    -DNV_TENSORRT_PATCH=0
)

# Set RPATH to find TensorRT libraries
set_target_properties(detect PROPERTIES
    BUILD_RPATH "${TensorRT_LIB_DIR}"
    INSTALL_RPATH "${TensorRT_LIB_DIR}"
)
