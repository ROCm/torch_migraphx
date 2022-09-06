# Torch-MIGraphX

Torch-MIGraphX integrates AMD's graph inference engine with the PyTorch ecosystem. It provides a `mgx_module` object that may be invoked in the same manner as any other torch module, but utilizes the MIGraphX inference engine under the hood.

## Installation
It is highly recommended to use the provided Dockerfile to create the environment requried to use Torch-MIGraphX. Build the container using:

```
docker build -t torch_migraphx .
```

Run the container:
```
sudo docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v=`pwd`:/code/torch_migraphx torch_migraphx
```

In the docker container, navigate to the py directory of this repository. Install pip prerequisites:

```
 pip install -r requirements.txt 
```
Install torch_migraphx:
```
python setup.py install
```
Add torch_migraphx to the pythonpath environment variable:
```
export PYTHONPATH=<path to torch_migraphx/py>:$PYTHONPATH
```

## Run Tests
Tests are written using pytest and are located in the Tests directory. Tests can be run using:
```
pytest
```

