FROM rocm/pytorch:rocm6.0.2_ubuntu22.04_py3.10_pytorch_2.1.2

ARG ROCM_PATH=/opt/rocm
ARG MIGRAPHX_BRANCH=master 
ARG GPU_ARCH

COPY . /workspace/torch_migraphx

# Install Dependencies: migraphx
RUN /workspace/torch_migraphx/tools/install_migraphx.sh ${MIGRAPHX_BRANCH} ${GPU_ARCH}

# Install torch_migraphx
RUN pip3 install pybind11-global
RUN cd /workspace/torch_migraphx/py && \
    export TORCH_CMAKE_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)") && \
    python -m pip install .

WORKDIR /workspace
ENV LD_LIBRARY_PATH=${ROCM_PATH}/lib
ENV PYTHONPATH=${ROCM_PATH}/lib