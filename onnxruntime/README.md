# Run example
```bash
docker build --progress=plain -t ort-cpu-example -f Dockerfile.ort-cpu-example .
docker run -it --rm ort-cpu-example cpu_example
```