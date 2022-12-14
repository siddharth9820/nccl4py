cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(nccl4py)

find_package(Torch REQUIRED)
find_package(CUDA 11.0 REQUIRED)
find_package(MPI REQUIRED)

include_directories(${MPI_INCLUDE_PATH})
# include_directories("/home/ssingh37/nccl/build/include")
# link_directories("/home/ssingh37/nccl/build/lib")
include_directories("/scratch/zt1/project/bhatele-lab/shared/nccl/build/include")
link_directories("/scratch/zt1/project/bhatele-lab/shared/nccl/build/lib")

add_library(nccl4py SHARED op.cpp)
target_link_libraries(nccl4py "${TORCH_LIBRARIES}")
target_link_libraries(nccl4py ${CUDA_LIBRARIES})
target_link_libraries(nccl4py ${MPI_LIBRARIES})
target_link_libraries(nccl4py "nccl")


if(MPI_COMPILE_FLAGS)
  set_target_properties(nccl4py PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(nccl4py PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

set_property(TARGET nccl4py PROPERTY CXX_STANDARD 14)
