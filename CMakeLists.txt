cmake_minimum_required(VERSION 3.16)
project(mnist)


find_package(OpenCV REQUIRED)
find_package(Tensorflow REQUIRED PATHS "/home/lola/tensorflow")
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(mnist mnist_infer.cpp )
target_link_libraries(mnist ${OpenCV_LIBS} ${TensorFlow_LIBRARIES})
set_property(TARGET mnist PROPERTY CXX_STANDARD 14)

