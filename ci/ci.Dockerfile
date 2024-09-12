ARG TAG=latest
FROM rocm/torch-migraphx-ci-ubuntu:${TAG}

COPY . /workspace/torch_migraphx

# Install torch_migraphx
RUN cd /workspace/torch_migraphx/py && \
    export TORCH_CMAKE_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)") && \
    python -m pip install .

WORKDIR /workspace