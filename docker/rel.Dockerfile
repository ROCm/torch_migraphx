## Use as reference for installing torch_migraphx on top of package manager installed deps
FROM ubuntu:22.04

WORKDIR /workspace

# Install rocm key
RUN apt-get update && apt-get install -y software-properties-common gnupg2 --no-install-recommends curl && \
    curl -sL http://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -

RUN sh -c 'echo deb [arch=amd64 trusted=yes] http://repo.radeon.com/rocm/apt/6.2/ jammy main > /etc/apt/sources.list.d/rocm.list'
RUN sh -c "echo 'Package: *\nPin: release o=repo.radeon.com\nPin-priority: 600' > /etc/apt/preferences.d/rocm-pin-600"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libjpeg-dev \
    python3-dev \
    python3-pip \
    cmake \
    hiprand-dev \
    hipblas \
    hipblaslt \
    hipcub \
    hipfft \
    hipsolver \
    hipsparse \
    rccl-dev \
    rocm-dev \
    rocthrust \
    migraphx-dev \
    git

RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.2/
RUN pip3 install pybind11-global

RUN git clone --single-branch https://github.com/ROCm/torch_migraphx.git \
    && cd torch_migraphx/py \
    && export TORCH_CMAKE_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)") \
    && python3 -m pip install .

