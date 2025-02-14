# if (APPLE)
#     set(DEPS tbb 
#         tbbmalloc tbbmalloc_proxy 
#         igl::core
#         suitesparseconfig metis cholmod amd camd ccolamd colamd
#         gmp mpfr boost_filesystem
#         )
# else()
#     set(DEPS boost_filesystem tbb 
#         tbbmalloc tbbmalloc_proxy 
#         mkl_intel_lp64 igl::core
#         mkl_sequential mkl_core 
#         suitesparseconfig metis cholmod amd camd ccolamd colamd
#         gmp mpfr
#         )
# endif()


file(GLOB HEADERS "include/*.h" "autodiff/*.h")
file(GLOB SOURCES "src/*.cpp" "autodiff/*.cpp")

find_package (Eigen3 REQUIRED)
include_directories( ${EIGEN3_INCLUDE_DIR} )
find_package(CHOLMOD REQUIRED)
include_directories(${CHOLMOD_INCLUDES})
find_package(TBB REQUIRED)
include_directories("/opt/homebrew/Cellar/tbb/2022.0.0/include")

add_executable(DiscreteShell ${HEADERS} ${SOURCES})
add_library(DiscreteShellLib STATIC ${HEADERS} ${SOURCES})

target_link_libraries(DiscreteShell ${CHOLMOD_LIBRARIES})
target_link_libraries(DiscreteShell TBB::tbb)

target_include_directories(DiscreteShell PUBLIC ../../Libs/spectra/include)
target_include_directories(DiscreteShell PUBLIC ../../Deps/libigl/include)
target_link_libraries(DiscreteShell ${DEPS} polyscope)
target_link_libraries(DiscreteShell Eigen3::Eigen)

target_include_directories(DiscreteShellLib PUBLIC ../../Libs/spectra/include)
target_include_directories(DiscreteShellLib PUBLIC ../../Deps/libigl/include)
target_link_libraries(DiscreteShellLib ${DEPS} polyscope)
target_link_libraries(DiscreteShell Eigen3::Eigen)
