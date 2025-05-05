# file(GLOB HEADERS "include/*.h" "autodiff/*.h")
# file(GLOB SOURCES "src/*.cpp" "autodiff/*.cpp")

# find_package (Eigen3 REQUIRED)
# include_directories( ${EIGEN3_INCLUDE_DIR} )
# # find_package(CHOLMOD REQUIRED)
# # include_directories(${CHOLMOD_INCLUDES})
# find_package(TBB REQUIRED)
# include_directories("/opt/homebrew/Cellar/tbb/2022.0.0/include")

# add_executable(DiscreteShell ${HEADERS} ${SOURCES})

# target_link_libraries(DiscreteShell TBB::tbb)
# target_link_libraries(DiscreteShell SuiteSparse::CHOLMOD)

# target_include_directories(DiscreteShell PUBLIC ../../Libs/spectra/include)
# target_include_directories(DiscreteShell PUBLIC ../../Deps/libigl/include)
# target_link_libraries(DiscreteShell ${DEPS} polyscope)
# target_link_libraries(DiscreteShell Eigen3::Eigen)

file(GLOB HEADERS "include/*.h" "autodiff/*.h")
file(GLOB SOURCES "src/*.cpp" "autodiff/*.cpp")

# find_package (Eigen3 REQUIRED)
# include_directories( ${EIGEN3_INCLUDE_DIR} )
# find_package(CHOLMOD REQUIRED)
# include_directories(${CHOLMOD_INCLUDES})
find_package(TBB REQUIRED)
find_package (SuiteSparse REQUIRED)
# include_directories("/opt/homebrew/Cellar/tbb/2022.0.0/include")

add_executable(DiscreteShell ${HEADERS} ${SOURCES})

target_link_libraries(DiscreteShell TBB::tbb)

target_include_directories(DiscreteShell PUBLIC ../../Libs/spectra/include)
target_include_directories(DiscreteShell PUBLIC ../../Deps/libigl/include)
target_link_libraries(DiscreteShell ${DEPS} polyscope)
target_link_libraries(DiscreteShell Eigen3::Eigen)
target_link_libraries(DiscreteShell ${SuiteSparse_CHOLMOD_LIBRARY})