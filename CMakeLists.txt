cmake_minimum_required(VERSION 2.8)

project(CUDA_DOS)

find_package(CUDA)
find_package(Boost COMPONENTS program_options REQUIRED)

list(APPEND CUDA_NVCC_FLAGS "-arch=sm_52;-std=c++11;-O2;-DVERBOSE")
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

CUDA_ADD_EXECUTABLE(DOS_CALC main.cu)
target_link_libraries(DOS_CALC ${CUDA_LIBRARIES} ${Boost_LIBRARIES})
