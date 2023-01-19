cmake_minimum_required(VERSION 3.5.0)
project(NN)

# SET(Boost_USE_STATIC_LIBS OFF)
# SET(Boost_USE_MULTITHREAD ON)

find_package(OpenCV REQUIRED)
find_package(Boost 1.80.0 REQUIRED COMPONENTS filesystem system thread regex PATHS /usr/local/boost_1_80_0/stage/lib/cmake/Boost-1.80.0)
include_directories(${BOOST_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIRS})
set(NEEDED_LIBS ${OpenCV_LIBS} ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY} ${Boost_REGEX_LIBRARY})

add_library(libs "")

# REMINDER: new scopes created... 
add_subdirectory(library)
add_subdirectory(source) 
