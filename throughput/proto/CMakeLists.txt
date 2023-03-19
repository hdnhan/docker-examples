find_package(protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(Threads)

set(PROTO_FILES tunnel.proto)
add_library(tunnel SHARED ${PROTO_FILES})
target_link_libraries(tunnel PUBLIC
    protobuf::libprotobuf
    gRPC::grpc
    gRPC::grpc++
)
target_include_directories(tunnel PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
get_target_property(grpc_cpp_plugin gRPC::grpc_cpp_plugin LOCATION)
set_target_properties(tunnel PROPERTIES POSITION_INDEPENDENT_CODE ON)
protobuf_generate(TARGET tunnel LANGUAGE cpp)
protobuf_generate(
    TARGET tunnel
    LANGUAGE grpc
    GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc
    PLUGIN protoc-gen-grpc=${grpc_cpp_plugin}
)
install(TARGETS tunnel DESTINATION ${CMAKE_INSTALL_LIBDIR})