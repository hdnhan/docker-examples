cmake_minimum_required(VERSION 3.5)
project(ort-example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra") # Optional
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_INSTALL_BINDIR ${CMAKE_INSTALL_PREFIX}/bin)
set(CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_PREFIX}/lib)

find_package(OpenCV REQUIRED)
# find_package(onnxruntime REQUIRED) # not working
find_library(ORT_LIBRARY NAME onnxruntime HINTS ${ORT_INSTALL_DIR}/lib REQUIRED)
set(ORT_INCLUDE_DIRS ${ORT_INSTALL_DIR}/include/onnxruntime/core/session)

add_executable(cpu_example main-cpu.cpp)
target_link_libraries(cpu_example ${OpenCV_LIBS} ${ORT_LIBRARY})
target_include_directories(cpu_example PRIVATE ${OpenCV_INCLUDE_DIRS} ${ORT_INCLUDE_DIRS})
install(TARGETS cpu_example DESTINATION ${CMAKE_INSTALL_BINDIR})

