# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sherin Joseph Roy. Licensed under the Apache License 2.0 (LICENSE in this folder).
cmake_minimum_required(VERSION 3.10)
project(vantage_bev_stitch CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(bev_stitch bev_stitch.cpp)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(bev_stitch PRIVATE OpenMP::OpenMP_CXX)
endif()
