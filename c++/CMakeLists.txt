cmake_minimum_required(VERSION 3.10)
project(detection LANGUAGES CXX CUDA)

# Detect platform
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" ARCH_NAME)
message(STATUS "System architecture: ${ARCH_NAME}")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Allow user to set CUDA path, fallback to default
if(NOT DEFINED ENV{CUDA_TOOLKIT_ROOT_DIR})
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
else()
    set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_TOOLKIT_ROOT_DIR})
endif()

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Suppress deprecated GPU architecture warnings
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

# Source files
set(HEADERS
    src/Detect.h
    src/preprocess.h
)

set(SOURCES
    src/main.cpp
    src/preprocess.cu
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/Detect.cpp")
    list(APPEND SOURCES src/Detect.cpp)
else()
    message(WARNING "src/Detect.cpp not found. Please check your sources.")
endif()

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# TensorRT detection
if(ARCH_NAME STREQUAL "aarch64")
    message(STATUS "Jetson detected - using TensorRT")
    set(TensorRT_INCLUDE_DIR "/usr/include/aarch64-linux-gnu")
    set(TensorRT_LIB_DIR "/usr/lib/aarch64-linux-gnu")
else()
    message(STATUS "x86_64 detected with TensorRT")
    set(TensorRT_INCLUDE_DIR "/usr/include/x86_64-linux-gnu")
    set(TensorRT_LIB_DIR "/usr/lib/x86_64-linux-gnu")
endif()

# Allow user to override TensorRT paths
set(TensorRT_INCLUDE_DIR ${TensorRT_INCLUDE_DIR} CACHE PATH "TensorRT include directory")
set(TensorRT_LIB_DIR ${TensorRT_LIB_DIR} CACHE PATH "TensorRT library directory")

# Check TensorRT header
if(NOT EXISTS "${TensorRT_INCLUDE_DIR}/NvInfer.h")
    message(FATAL_ERROR "TensorRT header (NvInfer.h) not found in ${TensorRT_INCLUDE_DIR}")
endif()

# TensorRT libraries
set(TensorRT_LIBS nvinfer nvinfer_plugin nvonnxparser)
foreach(lib ${TensorRT_LIBS})
    find_library(TensorRT_${lib}_LIBRARY NAMES ${lib} PATHS ${TensorRT_LIB_DIR} NO_DEFAULT_PATH)
    if(NOT TensorRT_${lib}_LIBRARY)
        message(FATAL_ERROR "TensorRT library ${lib} not found in ${TensorRT_LIB_DIR}")
    endif()
    list(APPEND TensorRT_LIBRARIES ${TensorRT_${lib}_LIBRARY})
endforeach()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TensorRT_INCLUDE_DIR}
)

# Static library
file(GLOB LIB_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/include/*.cpp"
    "${CMAKE_SOURCE_DIR}/include/*.h"
    "${CMAKE_SOURCE_DIR}/include/lanevision/*.cpp"
    "${CMAKE_SOURCE_DIR}/include/lanevision/*.h"
    "${CMAKE_SOURCE_DIR}/include/followdist/*.cpp"
    "${CMAKE_SOURCE_DIR}/include/followdist/*.h"
    "${CMAKE_SOURCE_DIR}/include/tensorrt/*.cpp"
    "${CMAKE_SOURCE_DIR}/include/tensorrt/*.h"
)
message(STATUS "LIB_SOURCES: ${LIB_SOURCES}") # Debug output
add_library(acc STATIC ${LIB_SOURCES})
target_include_directories(acc PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/lanevision
    ${CMAKE_SOURCE_DIR}/include/followdist
    ${CMAKE_SOURCE_DIR}/include/tensorrt
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        acc
        ${OpenCV_LIBS}
        ${TensorRT_LIBRARIES}
        ${CUDA_LIBRARIES}
        cudart
)

# CUDA architectures
if(ARCH_NAME STREQUAL "aarch64")
    set(CUDA_ARCH_LIST "53") # Jetson Nano
else()
    set(CUDA_ARCH_LIST "75;80;86") # Turing, Ampere, Ada Lovelace
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}"
    BUILD_RPATH "\$ORIGIN;${TensorRT_LIB_DIR};${CUDA_TOOLKIT_ROOT_DIR}/lib64"
    INSTALL_RPATH "\$ORIGIN;${TensorRT_LIB_DIR};${CUDA_TOOLKIT_ROOT_DIR}/lib64"
)

# Suppress deprecated declarations for TensorRT
if(ARCH_NAME STREQUAL "aarch64")
    target_compile_options(${PROJECT_NAME} PRIVATE -Wno-deprecated-declarations)
endif()