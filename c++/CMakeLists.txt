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

find_package(nlohmann_json REQUIRED)


# Gather all cpp/cu files EXCEPT main.cpp and test.cpp
file(GLOB_RECURSE ALL_SOURCES src/*.cpp src/*.cu)

# Define entry points
set(MAIN_ENTRY src/main.cpp)
set(TEST_ENTRY tests/test.cpp)

# Remove entry points from shared sources more explicitly
list(REMOVE_ITEM ALL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/${MAIN_ENTRY}
     ${MAIN_ENTRY})
set(SHARED_SOURCES ${ALL_SOURCES})

# Debug: Print what's in SHARED_SOURCES to verify
message(STATUS "SHARED_SOURCES contains: ${SHARED_SOURCES}")

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/Detect.cpp")
  # Only add if not already in SHARED_SOURCES
  list(FIND SHARED_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/Detect.cpp"
       DETECT_FOUND)
  if(DETECT_FOUND EQUAL -1)
    list(APPEND SHARED_SOURCES src/Detect.cpp)
  endif()
else()
  message(WARNING "src/Detect.cpp not found. Please check your sources.")
endif()

# Bytetrack library
include_directories(${PROJECT_SOURCE_DIR}/bytetrack/include)
file(GLOB_RECURSE SRCS01 ${PROJECT_SOURCE_DIR}/bytetrack/src/*.cpp)
add_library(bytetrack SHARED ${SRCS01})
target_link_libraries(bytetrack cudart nvinfer ${OpenCV_LIBS})

# Static library with CUDA support
file(GLOB LIB_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/include/*.h"
     "${CMAKE_SOURCE_DIR}/include/*.hpp")

include_directories(${CMAKE_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})

# Main executable
add_executable(${PROJECT_NAME} ${MAIN_ENTRY} ${SHARED_SOURCES})
target_link_libraries(
  ${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} ${TensorRT_LIBRARIES}
                          ${CUDA_LIBRARIES} cudart bytetrack nlohmann_json::nlohmann_json)
set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}"
             BUILD_RPATH
             "\$ORIGIN;${TensorRT_LIB_DIR};${CUDA_TOOLKIT_ROOT_DIR}/lib64"
             INSTALL_RPATH
             "\$ORIGIN;${TensorRT_LIB_DIR};${CUDA_TOOLKIT_ROOT_DIR}/lib64"
             RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Test executable
add_executable(test ${TEST_ENTRY} ${SHARED_SOURCES})
target_link_libraries(test PRIVATE ${OpenCV_LIBS} ${TensorRT_LIBRARIES}
                                   ${CUDA_LIBRARIES} cudart bytetrack nlohmann_json::nlohmann_json)
set_target_properties(
  test
  PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}"
             BUILD_RPATH
             "\$ORIGIN;${TensorRT_LIB_DIR};${CUDA_TOOLKIT_ROOT_DIR}/lib64"
             INSTALL_RPATH
             "\$ORIGIN;${TensorRT_LIB_DIR};${CUDA_TOOLKIT_ROOT_DIR}/lib64"
             RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
