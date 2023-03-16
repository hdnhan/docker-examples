add_executable(client client.cpp)
target_link_libraries(client PRIVATE tunnel)
install(TARGETS client DESTINATION ${CMAKE_INSTALL_BINDIR})