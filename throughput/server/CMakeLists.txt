find_package(OpenCV REQUIRED)
find_library(ORT_LIBRARY NAME onnxruntime HINTS ${ORT_INSTALL_DIR}/lib REQUIRED)
set(ORT_INCLUDE_DIRS ${ORT_INSTALL_DIR}/include/onnxruntime/core/session)


add_executable(server server.cpp)
target_link_libraries(server PRIVATE tunnel ${OpenCV_LIBS} ${ORT_LIBRARY})
target_include_directories(server PRIVATE ${OpenCV_INCLUDE_DIRS} ${ORT_INCLUDE_DIRS})
install(TARGETS server DESTINATION ${CMAKE_INSTALL_BINDIR})
