cmake_minimum_required(VERSION 3.0.2)
project(HomographyNet)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

set(CMAKE_PREFIX_PATH $ENV{HOME}/libtorch) # PyTorch laptop
# set(CMAKE_PREFIX_PATH $ENV{HOME}/.local/lib/python3.6/site-packages/torch) # PyTorch tx2

find_package(catkin REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 QUIET)
if (NOT OpenCV_FOUND)
    find_package(OpenCV 3 REQUIRED)
endif()
find_package(Boost REQUIRED COMPONENTS system filesystem thread date_time)

find_package(Torch REQUIRED) # PyTorch

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}") # PyTorch

catkin_package(
	INCLUDE_DIRS src
	LIBRARIES network_lib
)

###########
## Build ##
###########

# Include our header files
include_directories(
        src
        ${EIGEN3_INCLUDE_DIR}
        ${Boost_INCLUDE_DIRS}
        ${catkin_INCLUDE_DIRS}
)

# Set link libraries used by all binaries
list(APPEND thirdparty_libraries
        ${Boost_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${catkin_LIBRARIES}
	${TORCH_LIBRARIES}  # pytorch
)

add_library(network_lib SHARED
        src/HomographyNet.cpp
)

target_link_libraries(network_lib ${thirdparty_libraries})
target_include_directories(network_lib PUBLIC src)
set_property(TARGET network_lib PROPERTY CXX_STANDARD 14) # pytorch