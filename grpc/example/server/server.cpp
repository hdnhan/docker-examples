#include <grpcpp/grpcpp.h>
#include <signal.h>

#include <iostream>
#include <memory>
#include <string>

#include "tunnel.grpc.pb.h"

std::unique_ptr<grpc::Server> server;

// Signal handler function for SIGINT
void signalHandler(int signal) {
    std::cout << "Received SIGINT, shutting down server..." << std::endl;
    // Shut down the gRPC server gracefully
    server->Shutdown();
    // Exit the program
    exit(0);
}

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

    server = builder.BuildAndStart();
    std::cout << "Server listening on " << server_address << std::endl;

    // Set up the signal handler for SIGINT
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = signalHandler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

    server->Wait();
}

int main(/*int argc, char** argv*/) {
    RunServer();
    return 0;
}
