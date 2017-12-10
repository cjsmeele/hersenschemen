cmake_minimum_required(VERSION 3.5)
project(nn)

add_compile_options(-Wall -Wextra -pedantic)
set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES main.cc neural-network.cc neural-network.hh)
add_executable(nn ${SOURCE_FILES})
