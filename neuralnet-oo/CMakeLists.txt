cmake_minimum_required(VERSION 3.5)
project(nn)

add_compile_options(-Wall -Wextra -pedantic
                    -g0
                    -static -static-libgcc -static-libstdc++
                    -funroll-loops -march=native -mtune=native -Ofast
                    -std=c++14)

set(HEADER_FILES common.hh net.hh neuron.hh)
set(SOURCE_FILES main.cc ${HEADER_FILES})
add_executable(nn ${SOURCE_FILES})
