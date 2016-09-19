cmake_minimum_required(VERSION 3.0)

project(MRF)

set(SOURCE_FILES graph.cpp maxflow.cpp BP-S.cpp GCoptimization.cpp ICM.cpp LinkedBlockList.cpp MaxProdBP.cpp TRW-S.cpp mrf.cpp regions-maxprod.cpp)
set(HEADER_FILES block.h graph.h BP-S.h GCoptimization.h ICM.h LinkedBlockList.h MaxProdBP.h TRW-S.h energy.h mrf.h regions-new.h typeTruncatedQuadratic2D.h)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -O3 -DUSE_64_BIT_PTR_CAST")

add_library(MRF ${SOURCE_FILES} ${HEADER_FILES})

target_link_libraries(MRF glog)
