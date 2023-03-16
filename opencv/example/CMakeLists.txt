cmake_minimum_required(VERSION 3.5)
project(opencv-example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra") # Optional
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_INSTALL_BINDIR ${CMAKE_INSTALL_PREFIX}/bin)
set(CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_PREFIX}/lib)

find_package(OpenCV REQUIRED)

add_executable(cpu_example main-cpu.cpp)
target_link_libraries(cpu_example ${OpenCV_LIBS})
install(TARGETS cpu_example DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(cpu_example_transforms main-cpu-transforms.cpp)
target_link_libraries(cpu_example_transforms ${OpenCV_LIBS})
install(TARGETS cpu_example_transforms DESTINATION ${CMAKE_INSTALL_BINDIR})