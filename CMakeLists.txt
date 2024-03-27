cmake_minimum_required(VERSION 3.16)
project(svision_c)


find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


set(CMAKE_CXX_STANDARD 17)

add_executable(svision_c main.cpp)

target_link_libraries (svision_c ${PCL_LIBRARIES})

install(TARGETS svision_c RUNTIME DESTINATION bin)

