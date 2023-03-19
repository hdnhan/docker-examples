# Benchmark scenario
## Client
- Generate a 8uc3 image
- Make a request of running inference on the image on server

## Server
- Run inference on the image received from client.

# Setup
## Build images
```bash
# Build image for python client
docker build -t pyclient -f Dockerfile.client-python .

# Build image to generate onnx model
docker build -t genonnx -f Dockerfile.gen-onnx .

# Build image for python server
docker run -it --rm --network host pyserver python3 /server/server.py

# Build image for cpp server
docker build -t cppserver -f Dockerfile.server-cpp .
```

## Run benchmark
### First, start server
```bash
# Python
docker run -it --rm --network host pyserver python3 /server/server.py

# Or Cpp
docker run -it --rm --network host cppserver bin/server
```

### Star client
```bash
docker run -it --rm --network host pyclient python3 /client/client.py
```
