cmake_minimum_required(VERSION 2.8.3)
project(robot_vision_openvino)

## Compile as C++11, supported in ROS Kinetic and newer
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
set(CMAKE_C_FLAGS "-Wall -Wno-unused-result -Wno-unknown-pragmas -Wno-unused-variable -Wfatal-errors -fPIC ${CMAKE_C_FLAGS}")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (CMAKE_BUILD_TYPE STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE not defined, 'Release' will be used")
    set(CMAKE_BUILD_TYPE "Release")
endif()

if (NOT(BIN_FOLDER))
    string(TOLOWER ${CMAKE_SYSTEM_PROCESSOR} ARCH)
    if(ARCH STREQUAL "x86_64" OR ARCH STREQUAL "amd64") # Windows detects Intel's 64-bit CPU as AMD64
        set(ARCH intel64)
    elseif(ARCH STREQUAL "i386")
        set(ARCH ia32)
    endif()

    set (BIN_FOLDER ${ARCH})
endif()

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

# Find X11
message ( STATUS "Searching for X11..." )
find_package ( X11 REQUIRED )
if ( X11_FOUND )
  include_directories ( ${X11_INCLUDE_DIR} )
  link_libraries ( ${X11_LIBRARIES} )
  message ( STATUS " X11_INCLUDE_DIR: " ${X11_INCLUDE_DIR} )
  message ( STATUS " X11_LIBRARIES: " ${X11_LIBRARIES} )
endif ( X11_FOUND )

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

find_package(Boost REQUIRED 
 COMPONENTS 
  thread 
  system
)

find_package(ngraph REQUIRED PATHS "/opt/intel/openvino/deployment_tools/ngraph")
if (ngraph_FOUND)
  set (ngraph_INCLUDE_DIR /opt/intel/openvino/deployment_tools/ngraph/include)
  include_directories ( ${ngraph_INCLUDE_DIR} )
  link_libraries ( ${ngraph_LIBRARIES} )
  message ( STATUS " ngraph include DIRECTORY: " ${ngraph_INCLUDE_DIR})
  message ( STATUS " ngraph LIBRARIES: "${ngraph_LIBRARIES})
endif (ngraph_FOUND)

find_package(OpenCV REQUIRED
 COMPONENTS
  core
  highgui
  imgproc
)
include_directories(${OpenCV_INCLUDE_DIRS})


message ( STATUS " Searching for InferenceEngine... ")
find_package(InferenceEngine 2.0)
if (NOT InferenceEngine_FOUND)
    message(FATAL_ERROR "")
endif()
message ( STATUS " InferenceEngine_INCLUDE_DIRS: " ${InferenceEngine_INCLUDE_DIRS} )

find_package(catkin REQUIRED 
 COMPONENTS
  cv_bridge
  image_transport
  robot_control_msgs
  robot_vision_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf
  InferenceEngine
)
message (STATUS "Current DIRECTORY: "${CMAKE_CURRENT_SOURCE_DIR})

include_directories(
  include
  ${Boost_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include/samples
  ${CMAKE_CURRENT_SOURCE_DIR}/include/robot_vision_openvino/vino_openpose
  ${CMAKE_CURRENT_SOURCE_DIR}/include/robot_vision_openvino/vino_maskrcnn
  ${CMAKE_CURRENT_SOURCE_DIR}/include/robot_vision_openvino/vino_yolo
  ${CMAKE_CURRENT_SOURCE_DIR}/include/robot_vision_openvino/vino_interactive_face
  
  ${ngraph_INCLUDE_DIR}
  ${InferenceEngine_INCLUDE_DIRS}  
  ${InferenceEngine_INCLUDE_DIRS}/../samples
  ${InferenceEngine_DIR}/../src
)

set(OPENPOSE_LIB_FILES
  src/vino_openpose/peak.cpp
  src/vino_openpose/human_pose.cpp
  src/vino_openpose/render_human_pose.cpp
  src/vino_openpose/human_pose_estimator.cpp
  src/vino_openpose/openpose_ros.cpp
)

set(YOLO_LIB_FILES
  src/vino_yolo/vino_yolo.cpp
  src/vino_yolo/yolo_ros.cpp
)

set(MASKRCNN_LIB_FILES
  src/vino_maskrcnn/vino_maskrcnn.cpp
  src/vino_maskrcnn/maskrcnn_ros.cpp
) 

set(INTERACTIVE_FACE_LIB_FILES
  src/vino_interactive_face/detector.cpp
  src/vino_interactive_face/face.cpp
  src/vino_interactive_face/visualizer.cpp
  src/vino_interactive_face/ros_interface.cpp
) 

catkin_package(
  INCLUDE_DIRS 
   include
  LIBRARIES
   openpose_ros_lib
   yolo_ros_lib
  CATKIN_DEPENDS 
   cv_bridge 
   image_transport 
   roscpp 
   rospy 
   sensor_msgs 
   std_msgs 
   tf
  DEPENDS
   Boost
)


###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations

## Declare a C++ library
add_library(openpose_ros_lib SHARED
  ${OPENPOSE_LIB_FILES}
)

add_library(yolo_ros_lib SHARED
  ${YOLO_LIB_FILES}
)

add_library(maskrcnn_ros_lib SHARED
  ${MASKRCNN_LIB_FILES}
)

add_library(interactive_face_ros_lib SHARED
  ${INTERACTIVE_FACE_LIB_FILES}
)


## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(openpose_ros src/human_pose_estimate_main.cpp)

add_executable(yolo_ros src/object_detection_main.cpp)

add_executable(maskrcnn_ros src/semantic_segmentation_main.cpp)

add_executable(interactive_face_ros src/interactive_face_recognition_main.cpp)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(openpose_ros_lib
  m
  pthread
  stdc++
  ${Boost_LIBRARIED}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

target_link_libraries(openpose_ros
  openpose_ros_lib
)

target_link_libraries(yolo_ros_lib
  m
  pthread
  stdc++
  ${ngraph_LIBS}
  ${Boost_LIBRARIED}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

target_link_libraries(yolo_ros
  yolo_ros_lib
)

target_link_libraries(maskrcnn_ros_lib
  m
  pthread
  stdc++
  ${ngraph_LIBS}
  ${Boost_LIBRARIED}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

target_link_libraries(maskrcnn_ros
  maskrcnn_ros_lib
)

target_link_libraries(interactive_face_ros_lib
  m
  pthread
  stdc++
  ${ngraph_LIBS}
  ${Boost_LIBRARIED}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

target_link_libraries(interactive_face_ros
  interactive_face_ros_lib
)


#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_robot_vision_openvino.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
