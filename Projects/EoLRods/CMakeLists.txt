# if (APPLE)
#     set(DEPS  
#         igl::core
#         )
# else()
#     set(DEPS boost_filesystem tbb 
#         tbbmalloc tbbmalloc_proxy 
#         mkl_intel_lp64 
#         mkl_sequential mkl_core 
#         suitesparseconfig metis cholmod amd camd ccolamd colamd
#         gmp mpfr
#     )
# endif()

file(GLOB HEADERS "include/*.h" "autodiff/*.h")
file(GLOB SOURCES "src/*.cpp" "autodiff/*.cpp")
find_package(CHOLMOD REQUIRED)
include_directories(${CHOLMOD_INCLUDES})
find_package(TBB REQUIRED)
include_directories("/opt/homebrew/Cellar/tbb/2022.0.0/include")

add_executable(EoLRods ${HEADERS} ${SOURCES})
target_link_libraries(EoLRods ${CHOLMOD_LIBRARIES})
target_link_libraries(EoLRods TBB::tbb)

target_include_directories(EoLRods PUBLIC ../../Libs/spectra/include)
target_include_directories(EoLRods PUBLIC ../../Deps/libigl/include)
target_link_libraries(EoLRods ${DEPS} polyscope)
