project(lippolc)
cmake_minimum_required(VERSION 3.12)

find_package(OpenMP)

set(CMAKE_CXX_STANDARD 17)

add_compile_options(-O0 -Wall)

include_directories(src/core)

# add_executable(example src/examples/example.cpp)
# add_executable(example_without_fmcd src/examples/example_without_fmcd.cpp)
# add_executable(example_bulk_load src/examples/example_bulk_load.cpp)
add_executable(example_mt src/examples/example_mt.cpp)

target_link_libraries(example_mt PUBLIC OpenMP::OpenMP_CXX tbb)

################################
######## For benchmark ######### (not used yet)
################################
# include(FetchContent)
# include(ExternalProject)

# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)

# include(CheckCXXCompilerFlag)
# check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
# if (COMPILER_SUPPORTS_MARCH_NATIVE)
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -lnuma -lrt -lpthread -mrtm -msse4.1 -mavx2 -O3 -g -march=native -w -Wfatal-errors")
# endif ()

# add_compile_options(-Wall -fmax-errors=5 -faligned-new -march=native -mtune=native -mavx2)

# # microbench
# add_executable(microbench ${CMAKE_CURRENT_SOURCE_DIR}/src/benchmark/microbench.cpp)
# string(TOLOWER ${CMAKE_BUILD_TYPE} _type)
# if (_type STREQUAL release)
#     set(default_build_type "Release")
#     message(STATUS "Setting build type to '${default_build_type}' ")
#     target_compile_definitions(microbench PRIVATE NDEBUGGING)
# else()
#     message(STATUS "Setting build type to '${default_build_type}' ")
# endif()

# target_link_libraries(microbench PRIVATE -lpthread -ltbb)
# target_link_libraries(microbench PUBLIC OpenMP::OpenMP_CXX)