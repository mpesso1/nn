

target_include_directories(libs PUBLIC
    ${CMAKE_CURRENT_LIST_DIR})

target_sources(libs
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/base.cpp
    ${CMAKE_CURRENT_LIST_DIR}/base.hpp)

add_subdirectory(communication)
