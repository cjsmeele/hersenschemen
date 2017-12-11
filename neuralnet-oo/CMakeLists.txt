cmake_minimum_required(VERSION 3.5)
project(nn)

add_compile_options(-Wall -Wextra -pedantic -g3 -O3 -std=c++14)

set(SOURCE_FILES main.cc)
add_executable(nn ${SOURCE_FILES})
