#include <grpcpp/grpcpp.h>

#include <iostream>
#include <memory>
#include <string>

#include "tunnel.grpc.pb.h"

class BenchServiceImpl final : public tunnel::BenchService::Service {
    grpc::Status StreamData(
        grpc::ServerContext* context,
        grpc::ServerReaderWriter<tunnel::Response, tunnel::Request>* stream) override {
        tunnel::Request request;
        while (stream->Read(&request)) {
            tunnel::Response response;
            response.set_data(request.data());
            stream->Write(response);
        }
        return grpc::Status::OK;
    }
};

void RunServer() {
    std::string server_address("0.0.0.0:50051");
    BenchServiceImpl service;

    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    std::cout << "Server listening on " << server_address << std::endl;
    server->Wait();
}

int main(/*int argc, char** argv*/) {
    RunServer();
    return 0;
}
