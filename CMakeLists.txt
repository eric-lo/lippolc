project(lippolc)
cmake_minimum_required(VERSION 3.12)

include(FetchContent)
include(ExternalProject)

find_package(OpenMP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if (COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -lnuma -lrt -lpthread -mrtm -msse4.1 -mavx2 -O3 -g -march=native -w -Wfatal-errors")
endif ()

# Set a default build type if none was specified
# https://blog.kitware.com/cmake-and-the-default-build-type/
set(default_build_type "Release")
if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
    set(default_build_type "Debug")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
            STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
            "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

add_compile_options(-Wall -fmax-errors=5 -faligned-new -march=native -mtune=native -mavx2)

include_directories(src/core)

add_executable(example src/examples/example.cpp)
add_executable(example_without_fmcd src/examples/example_without_fmcd.cpp)
add_executable(example_bulk_load src/examples/example_bulk_load.cpp)

# microbench
add_executable(microbench ${CMAKE_CURRENT_SOURCE_DIR}/src/benchmark/microbench.cpp)
string(TOLOWER ${CMAKE_BUILD_TYPE} _type)
if (_type STREQUAL release)
    set(default_build_type "Release")
    message(STATUS "Setting build type to '${default_build_type}' ")
    target_compile_definitions(microbench PRIVATE NDEBUGGING)
else()
    message(STATUS "Setting build type to '${default_build_type}' ")
endif()

target_link_libraries(microbench PRIVATE -lpthread -ltbb)
target_link_libraries(microbench PUBLIC OpenMP::OpenMP_CXX)