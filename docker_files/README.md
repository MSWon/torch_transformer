## build docker file

### horovod 

```
$ cd horovod
$ docker build --tag pytorch_hvd:0.0.1 .
```

### TensorRT 

- Download `TensorRT-8.2.3.0` from https://developer.nvidia.com/nvidia-tensorrt-download

```
$ cd tensorrt
$ docker build --tag torch2trt:0.0.1 .
```