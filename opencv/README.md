# Run example
```bash
docker build --progress=plain -t opencv-cpu-example -f Dockerfile.opencv-cpu-example .
docker run -it --rm opencv-cpu-example cpu_example
docker run -it --rm opencv-cpu-example cpu_example_transforms
```