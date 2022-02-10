project(lippolc)
cmake_minimum_required(VERSION 3.12)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(OpenMP)
find_package(TBB REQUIRED)

set(CMAKE_CXX_STANDARD 17)

add_compile_options(-O0 -Wall)

include_directories(src/core)
include_directories(${TBB_INCLUDE_DIRS})

# add_executable(example src/examples/example.cpp)
# add_executable(example_without_fmcd src/examples/example_without_fmcd.cpp)
# add_executable(example_bulk_load src/examples/example_bulk_load.cpp)
add_executable(example_mt src/examples/example_mt.cpp)

target_link_libraries(example_mt PUBLIC OpenMP::OpenMP_CXX ${TBB_LIBRARIES})
