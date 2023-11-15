# Build Torch-MIGraphX Environments

The suggested nightly build is provided in the main project directory. This directory provides additional environment setups such as for development.

## Development
This is essentially identical to the nightly environment, however `dev.Dockerfile` installs a development version of MIGraphX. Torch-MIGraphX is left out for the user to install in development mode.

```
#1. Build dev.Dockerfile:
docker build -t torch_migraphx_dev -f ./docker/dev.Dockerfile .

#2. Start container with mounted torch_migraphx directory:
sudo docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v=`pwd`:/workspace/torch_migraphx torch_migraphx_dev

#3. Install torch_migraphx in dev mode:
cd /workspace/torch_migraphx/py
export TORCH_CMAKE_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)") 
pip install -e .
```