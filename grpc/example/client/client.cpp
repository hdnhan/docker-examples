#include <grpcpp/grpcpp.h>

#include <iostream>
#include <memory>
#include <string>

#include "tunnel.grpc.pb.h"

class BenchClient {
   public:
    BenchClient(std::shared_ptr<grpc::Channel> channel)
        : stub_(tunnel::BenchService::NewStub(channel)) {}

    void StreamData() {
        grpc::ClientContext context;
        std::shared_ptr<grpc::ClientReaderWriter<tunnel::Request, tunnel::Response>> stream(
            stub_->StreamData(&context));

        for (int i = 0; i < 1000; i++) {
            tunnel::Request request;
            request.set_data("Hello, server! This is request #" + std::to_string(i));
            stream->Write(request);

            tunnel::Response response;
            while (stream->Read(&response)) {
                std::cout << "Received response: " << response.data() << std::endl;
                break;
            }
        }

        stream->WritesDone();
        grpc::Status status = stream->Finish();
        if (status.ok()) {
            std::cout << "StreamData rpc succeeded." << std::endl;
        } else {
            std::cout << "StreamData rpc failed: " << status.error_message() << std::endl;
        }
    }

   private:
    std::unique_ptr<tunnel::BenchService::Stub> stub_;
};

int main(/*int argc, char** argv*/) {
    BenchClient client(grpc::CreateChannel("localhost:50051", grpc::InsecureChannelCredentials()));
    client.StreamData();

    return 0;
}
