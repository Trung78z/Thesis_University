cmake_minimum_required(VERSION 3.10.2)
project(detection LANGUAGES CXX CUDA)

# Log TensorRT version
execute_process(
    COMMAND bash -c "grep '#define NV_TENSORRT_MAJOR' ${TensorRT_INCLUDE_DIR}/NvInferVersion.h | awk '{print $3}'"
    OUTPUT_VARIABLE TRT_MAJOR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND bash -c "grep '#define NV_TENSORRT_MINOR' ${TensorRT_INCLUDE_DIR}/NvInferVersion.h | awk '{print $3}'"
    OUTPUT_VARIABLE TRT_MINOR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND bash -c "grep '#define NV_TENSORRT_PATCH' ${TensorRT_INCLUDE_DIR}/NvInferVersion.h | awk '{print $3}'"
    OUTPUT_VARIABLE TRT_PATCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "TensorRT version detected: ${TRT_MAJOR}.${TRT_MINOR}.${TRT_PATCH}")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define CUDA path
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda" CACHE PATH "Path to CUDA toolkit")
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Add headers (optional for IDE support)
set(HEADERS
    src/Detect.h
    src/macros.h
    src/logging.h
    src/cuda_utils.h
    src/preprocess.h
    src/common.h
)

# Add source files
set(SOURCES
    main.cpp
    src/preprocess.cu
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/Detect.cpp")
    list(APPEND SOURCES src/Detect.cpp)
else()
    message(WARNING "src/Detect.cpp not found. Please check your sources.")
endif()

# OpenCV
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found!")
endif()
include_directories(${OpenCV_INCLUDE_DIRS})

# TensorRT
set(TensorRT_INCLUDE_DIR "/usr/include/aarch64-linux-gnu" CACHE PATH "TensorRT include path")
set(TensorRT_LIB_DIR "/usr/lib/aarch64-linux-gnu" CACHE PATH "TensorRT library path")

# Check header
if (NOT EXISTS "${TensorRT_INCLUDE_DIR}/NvInfer.h")
    message(FATAL_ERROR "TensorRT header (NvInfer.h) not found in ${TensorRT_INCLUDE_DIR}")
endif()

# TensorRT libraries to link
set(TensorRT_LIB_NAMES
    nvinfer
    nvinfer_plugin
    nvonnxparser
)

# Resolve actual library paths
set(TensorRT_LINK_LIBS "")
foreach(lib ${TensorRT_LIB_NAMES})
    find_library(${lib}_LIBRARY NAMES ${lib} PATHS ${TensorRT_LIB_DIR})
    if (NOT ${lib}_LIBRARY)
        message(FATAL_ERROR "TensorRT library ${lib} not found in ${TensorRT_LIB_DIR}")
    endif()
    list(APPEND TensorRT_LINK_LIBS ${${lib}_LIBRARY})
endforeach()

# Include dirs
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TensorRT_INCLUDE_DIR}
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${TensorRT_LINK_LIBS}
    ${CUDA_LIBRARIES}
    cudart
    pthread
    dl
)

# Optional: CUDA architecture (adjust based on your GPU)
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_ARCHITECTURES "60;70;75;80;86"
)

# Optional: RPATH for runtime
set_target_properties(${PROJECT_NAME} PROPERTIES
    BUILD_RPATH "${TensorRT_LIB_DIR};${CUDA_TOOLKIT_ROOT_DIR}/lib64"
    INSTALL_RPATH "${TensorRT_LIB_DIR};${CUDA_TOOLKIT_ROOT_DIR}/lib64"
)
