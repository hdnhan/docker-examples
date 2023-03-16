# Run example
```bash
docker build --progress=plain -t grpc-example -f Dockerfile.grpc-example .
docker run -it --rm --network host grpc-example server
docker run -it --rm --network host grpc-example client
```