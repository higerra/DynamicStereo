cmake_minimum_required(VERSION 3.0)

project(SubspaceStab)

find_package(OpenCV REQUIRED)
find_package(gflags REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -Wno-deprecated-declarations -Wno-deprecated-register")
set(CMAKE_BUILD_TYPE Release)

include_directories(/usr/local/include/eigen3 /usr/include/suitesparse)
link_directories(/usr/local/lib /usr/lib/x86_64-linux-gnu/)

unset(OPENMP_FOUND)
find_package(OpenMP QUIET)
if(OPENMP_FOUND)
  message("-- Found OpenMP")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_library(SubspaceStab tracking.h tracking.cpp
        factorization.cpp factorization.h
        warping.h warping.cpp
        subspacestab.h subspacestab.cpp)
target_link_libraries(SubspaceStab ${OpenCV_LIBS} glog gflags spqr cholmod)

add_executable(SubspaceStab_exe main.cpp)
target_link_libraries(SubspaceStab_exe SubspaceStab)


