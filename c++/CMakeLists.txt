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

# Source files
set(HEADERS
    src/Detect.h
    src/logging.h
    src/preprocess.h
    src/common.h
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
    message(STATUS "Jetson detected - using TensorRT < 8.5")
    set(TensorRT_INCLUDE_DIR "/usr/include/aarch64-linux-gnu")
    set(TensorRT_LIB_DIR "/usr/lib/aarch64-linux-gnu")
else()
    message(STATUS "Ubuntu detected with TensorRT")
    set(TensorRT_INCLUDE_DIR "/usr/include/x86_64-linux-gnu")
    set(TensorRT_LIB_DIR "/usr/lib/x86_64-linux-gnu")
endif()
# add_compile_options(-Wno-deprecated-declarations)

# Check header
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
endforeach()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TensorRT_INCLUDE_DIR}
)
file(GLOB_RECURSE LIB_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/include/*.cpp"
    "${CMAKE_SOURCE_DIR}/include/*.h"
)

add_library(libfollowdist STATIC ${LIB_SOURCES})
target_include_directories(libfollowdist PUBLIC ${CMAKE_SOURCE_DIR}/include)
# Create executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
# Link libraries
target_link_libraries(${PROJECT_NAME}
    PRIVATE libfollowdist
    ${OpenCV_LIBS}
    ${TensorRT_LIBS}
    ${CUDA_LIBRARIES}
    cudart
)

# CUDA architectures
if(ARCH_NAME STREQUAL "aarch64")
    # Jetson Nano = compute 53 (sm_53)
    set(CUDA_ARCH_LIST "53")
else()
    set(CUDA_ARCH_LIST "60;70;75;80;86")
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}"
    BUILD_RPATH "\$ORIGIN;${TensorRT_LIB_DIR};${CUDA_TOOLKIT_ROOT_DIR}/lib64"
    INSTALL_RPATH "\$ORIGIN;${TensorRT_LIB_DIR};${CUDA_TOOLKIT_ROOT_DIR}/lib64"
)