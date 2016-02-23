cmake_minimum_required(VERSION 3.0)
project(OpticalFlow)

set(HEADER_FILES GaussianPyramid.h Image.h ImageIO.h ImageProcessing.h Matrix.h NoiseModel.h OpticalFlow.h Stochastic.h Vector.h)
set(SOURCE_FILES GaussianPyramid.cpp OpticalFlow.cpp Stochastic.cpp)
add_library(OpticalFlow ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(OpticalFlow ${OpenCV_LIBS})
