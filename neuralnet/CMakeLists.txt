cmake_minimum_required(VERSION 3.5)
project(nn)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES main.cpp neural-network.cc neural-network.hh)
add_executable(4 ${SOURCE_FILES})
