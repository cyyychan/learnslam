cmake_minimum_required( VERSION 2.8 )
project(feature_extraction)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package( OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable( feature_extraction feature_extraction.cpp)
target_link_libraries(feature_extraction ${OpenCV_LIBS})