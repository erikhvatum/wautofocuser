# The MIT License (MIT)
#
# Copyright (c) 2014 WUSTL ZPLAB
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Authors: Erik Hvatum <ice.rikh@gmail.com>

# CMake 2.8.11 introduced the target_link_libraries command, a huge time saver
cmake_minimum_required(VERSION 2.8.11)

project(wautofocus)
set(MAJOR_VERSION 0)
set(MINOR_VERSION 1)
set(PACKAGE wautofocus)
set(PACKAGE_VERSION ${MAJOR_VERSION}.${MINOR_VERSION})

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON CACHE BOOL "")
set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "")
if(NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC" CACHE STRING "")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC" CACHE STRING "")
endif()

find_package(OpenCV 3 REQUIRED)
add_executable(wautofocuser main.cpp)
target_link_libraries(wautofocuser ${OpenCV_LIBS})
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
