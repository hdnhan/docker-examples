# Run example
```bash
docker build -t genonnx -f Dockerfile.gen-onnx .
docker build --progress=plain -t ort-cpu-example -f Dockerfile.ort-cpu-example .
docker run -it --rm ort-cpu-example
```

Dockerfile directly from ONNNRuntime https://github.com/microsoft/onnxruntime/tree/main/dockerfiles
Different execution providers: https://onnxruntime.ai/docs/build/eps.html#execution-provider-shared-libraries