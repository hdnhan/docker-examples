add_executable(server server.cpp)
target_link_libraries(server PRIVATE tunnel)
install(TARGETS server DESTINATION ${CMAKE_INSTALL_BINDIR})