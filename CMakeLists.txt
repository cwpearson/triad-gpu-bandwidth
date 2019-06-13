cmake_minimum_required(VERSION 3.13)

include("cmake/HunterGate.cmake")
HunterGate(
    URL "https://github.com/ruslo/hunter/archive/v0.23.186.tar.gz"
    SHA1 "b215dc4d88cbe6cc74b5cc9fa9cdd7058775c26b"
)

project(um LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)

# Set a default build type if none was specified
set(default_build_type "Release")
# if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
#   set(default_build_type "Debug")
# endif()
 
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

hunter_add_package(cxxopts)
find_package(cxxopts CONFIG REQUIRED)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "-stdlib=libc++")
  set(CMAKE_CUDA_FLAGS "-Xcompiler=-stdlib=libc++")
  message(STATUS "added some clang flags")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
endif()

# set C++ and CUDA standards for all targets
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

add_executable(triad triad.cu cache.cu test_system_allocator.cu)
target_link_libraries(triad PUBLIC cxxopts::cxxopts)


if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_link_libraries(triad PUBLIC c++)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
endif()
