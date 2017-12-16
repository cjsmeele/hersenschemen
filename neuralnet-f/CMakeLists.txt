cmake_minimum_required(VERSION 3.5)
project(nn)

add_compile_options(-Wall -Wextra -pedantic-errors
                    -g3
                    -ftemplate-backtrace-limit=0
                    -static -static-libgcc -static-libstdc++
                    -funroll-loops -march=native -mtune=native -Ofast
                    -std=c++17)

set(SOURCE_FILES main.cc)
add_executable(nn ${SOURCE_FILES})
