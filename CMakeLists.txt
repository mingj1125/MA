cmake_minimum_required(VERSION 3.20)
add_definitions(-w)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message( FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt." )
endif()
set(CMAKE_CXX_COMPILER g++-14)
# set(CMAKE_C_COMPILER gcc-14)
# set(CMAKE_CXX_COMPILER clang++)
# set(CMAKE_C_COMPILER clang)
project(WuKong2024 CXX)

message(STATUS "${CMAKE_BUILD_TYPE} Build")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake)

set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")
set(CMAKE_CXX_STANDARD 17)
set(MKL_THREADING "TBB")

find_package (Eigen3 3.4.0 REQUIRED)
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
find_package(Ceres REQUIRED)
find_package (SuiteSparse REQUIRED)

# set(CMAKE_CXX_FLAGS "-stdlib=libstdc++ ${CMAKE_CXX_FLAGS} -fPIC -O3 -Wno-unused-variable -Wno-unused-but-set-variable -Wsign-compare -Wreorder")

include_directories( ${EIGEN3_INCLUDE_DIR})

add_subdirectory("Deps/libigl")
add_subdirectory("Deps/polyscope")

include(libigl)
igl_include(predicates)

find_package(CGAL REQUIRED COMPONENTS Core)
  
if (NOT APPLE)
    add_definitions(-DEIGEN_USE_MKL_ALL)
    add_definitions(-DMKL_LP64)

    set(MKLROOT "/opt/intel/oneapi/mkl/2024.1")
    include_directories(${MKLROOT}/include)
    link_directories(${MKLROOT}/lib)
endif()

include_directories(/usr/local/include/suitesparse)
include_directories(${SUITESPARSE_INCLUDE_DIRS})
include_directories(/usr/local/include/)
link_directories(/usr/local/lib)

include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(Libs)
add_subdirectory(Projects)


message("**************************************************")
message("C++ Cmake Flags: ${CMAKE_CXX_FLAGS}")
message("**************************************************")
