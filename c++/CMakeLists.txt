
cmake_minimum_required(VERSION 3.10)
project(detection LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Find dependencies
include(FindCUDAConfig)
include(FindOpenCVConfig)
include(FindTensorRTConfig)

# Include Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Source files
set(HEADERS src/preprocess.h)

set(SOURCES src/main.cpp src/preprocess.cu src/utils.cpp)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/Detect.cpp")
  list(APPEND SOURCES src/Detect.cpp)
else()
  message(WARNING "src/Detect.cpp not found. Please check your sources.")
endif()

# Bytetrack library
include_directories(${PROJECT_SOURCE_DIR}/bytetrack/include)
file(GLOB_RECURSE SRCS01 ${PROJECT_SOURCE_DIR}/bytetrack/src/*.cpp)
add_library(bytetrack SHARED ${SRCS01})
target_link_libraries(bytetrack cudart nvinfer ${OpenCV_LIBS})

# Static library with CUDA support
file(
  GLOB
  LIB_SOURCES
  CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/include/*.cpp"
  "${CMAKE_SOURCE_DIR}/include/*.h"
  "${CMAKE_SOURCE_DIR}/include/lanevision/*.cpp"
  "${CMAKE_SOURCE_DIR}/include/lanevision/*.h"
  "${CMAKE_SOURCE_DIR}/include/followdist/*.cpp"
  "${CMAKE_SOURCE_DIR}/include/followdist/*.h"
  "${CMAKE_SOURCE_DIR}/include/tensorrt/*.cpp"
  "${CMAKE_SOURCE_DIR}/include/tensorrt/*.h")

# CUDA-enabled static lib
add_library(acc STATIC ${LIB_SOURCES})
target_include_directories(
  acc
  PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/include/lanevision
         ${CMAKE_SOURCE_DIR}/include/followdist
         ${CMAKE_SOURCE_DIR}/include/tensorrt ${OpenCV_INCLUDE_DIRS})

# Executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(
  ${PROJECT_NAME} PRIVATE acc ${OpenCV_LIBS} ${TensorRT_LIBRARIES}
                          ${CUDA_LIBRARIES} cudart bytetrack)

set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}"
             BUILD_RPATH
             "\$ORIGIN;${TensorRT_LIB_DIR};${CUDA_TOOLKIT_ROOT_DIR}/lib64"
             INSTALL_RPATH
             "\$ORIGIN;${TensorRT_LIB_DIR};${CUDA_TOOLKIT_ROOT_DIR}/lib64")

# Set output directory for 'main' binary
set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                 ${CMAKE_SOURCE_DIR}/bin)