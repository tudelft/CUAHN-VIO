cmake_minimum_required(VERSION 2.8.8)

# Project name
project(cuahn)

# set(CMAKE_PREFIX_PATH $ENV{HOME}/libtorch) # PyTorch laptop # NOTE! Comment this line after the first (failed) catkin build attempt
# set(CMAKE_PREFIX_PATH $ENV{HOME}/.local/lib/python3.6/site-packages/torch) # PyTorch tx2

# Find catkin (the ROS build system)
find_package(catkin QUIET COMPONENTS roscpp rosbag tf std_msgs geometry_msgs sensor_msgs nav_msgs visualization_msgs cv_bridge ov_core HomographyNet)

# Include libraries (if we don't have opencv 4, then fallback to opencv 3)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 QUIET)
if (NOT OpenCV_FOUND)
    find_package(OpenCV 3 REQUIRED)
endif()
find_package(Boost REQUIRED COMPONENTS system filesystem thread date_time)
find_package(Torch REQUIRED) # PyTorch    

# Display message to user
message(WARNING "OPENCV VERSION: " ${OpenCV_VERSION})
message(WARNING "BOOST VERSION: " ${Boost_VERSION})

# If we will compile with aruco support
option(ENABLE_ARUCO_TAGS "Enable or disable aruco tag (disable if no contrib modules)" ON)
if (NOT ENABLE_ARUCO_TAGS)
    add_definitions(-DENABLE_ARUCO_TAGS=0)
    message(WARNING "DISABLING ARUCOTAG TRACKING!")
else()
    add_definitions(-DENABLE_ARUCO_TAGS=1)
endif()

# Describe catkin project
option(ENABLE_CATKIN_ROS "Enable or disable building with ROS (if it is found)" ON)
if (catkin_FOUND AND ENABLE_CATKIN_ROS)
    add_definitions(-DROS_AVAILABLE=1)
    catkin_package(
            CATKIN_DEPENDS roscpp rosbag tf std_msgs geometry_msgs sensor_msgs nav_msgs visualization_msgs cv_bridge ov_core HomographyNet
            INCLUDE_DIRS src
            LIBRARIES cuahn_lib
    )
else()
    add_definitions(-DROS_AVAILABLE=0)
    message(WARNING "BUILDING WITHOUT ROS!")
endif()

# Try to compile with c++11
# http://stackoverflow.com/a/25836953
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# Enable compile optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")

# Enable debug flags (use if you want to debug in gdb)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -Wall -Wuninitialized -Wmaybe-uninitialized -fno-omit-frame-pointer")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}") # PyTorch (not required in catkin build)
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
        ${TORCH_LIBRARIES}  # PyTorch
)

# If we are not building with ROS then we need to manually link to its headers
# This isn't that elegant of a way, but this at least allows for building without ROS
# See this stackoverflow answer: https://stackoverflow.com/a/11217008/7718197
if (NOT catkin_FOUND OR NOT ENABLE_CATKIN_ROS)
    message(WARNING "MANUALLY LINKING TO OV_CORE LIBRARY....")
    include_directories(${ov_core_SOURCE_DIR}/src/)
    list(APPEND thirdparty_libraries ov_core_lib)

    message(WARNING "MANUALLY LINKING TO NETWORK_LIB LIBRARY....")
    include_directories(${network_lib}/src/)
    list(APPEND thirdparty_libraries network_lib)
endif()

##################################################
# Make the core library
##################################################
list(APPEND library_source_files
        src/sim/Simulator.cpp
        src/state/State.cpp
        src/state/StateHelper.cpp
        src/state/Propagator.cpp
        src/core/VioManager.cpp
        src/update/UpdaterHNet.cpp
)
if (catkin_FOUND AND ENABLE_CATKIN_ROS)
    list(APPEND library_source_files
        src/core/RosVisualizer.cpp
    )
endif()
add_library(cuahn_lib SHARED ${library_source_files})
target_link_libraries(cuahn_lib ${thirdparty_libraries})
set_property(TARGET cuahn_lib PROPERTY CXX_STANDARD 14) # PyTorch
target_include_directories(cuahn_lib PUBLIC src)


##################################################
# Make binary files!
##################################################


if (catkin_FOUND AND ENABLE_CATKIN_ROS)

    add_executable(run_subscribe_cuahn src/ros_subscribe_cuahn.cpp)
    target_link_libraries(run_subscribe_cuahn cuahn_lib ${thirdparty_libraries})

    set_property(TARGET run_subscribe_cuahn PROPERTY CXX_STANDARD 14) # PyTorch

else()

endif()
