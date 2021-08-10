cmake_minimum_required(VERSION 3.16)
project(mnist C CXX)


find_package(Tensorflow REQUIRED PATHS "/home/lola/tensorflow")

set(CMAKE_CXX_STANDARD 11)
add_executable(mnist
  mnist_infer.cpp
)
target_link_libraries(mnist ${TensorFlow_LIBRARIES})