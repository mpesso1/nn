cmake_minimum_required(VERSION 3.5.0)
project(NN)

# add_subdirectory(library)
add_library(base library/base.cpp library/base.h)
target_include_directories(base PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/library)

# add_subdirectory(source)
add_executable(run source/run.cpp)

target_link_libraries(run PUBLIC base)

# target_include_directories()
