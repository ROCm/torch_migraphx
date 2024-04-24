FROM rocm/pytorch:rocm6.0.2_ubuntu22.04_py3.10_pytorch_2.1.2

ARG ROCM_PATH=/opt/rocm
ARG MIGRAPHX_BRANCH=master 
ARG GPU_ARCH="gfx900;gfx906;gfx908;gfx90a;gfx1030;gfx1100;gfx1101;gfx1102;gfx940;gfx941;gfx942"

COPY . /workspace/torch_migraphx

# Install Dependencies: MIGraphX
# Install rbuild
RUN pip install https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz
# update rocm-cmake
RUN git clone https://github.com/RadeonOpenCompute/rocm-cmake.git \
    && cd rocm-cmake  \
    && git checkout 5a34e72d9f113eb5d028e740c2def1f944619595 \
    && mkdir build && cd build \
    && cmake .. && cmake --build . --target install
# install migraphx from source
RUN git clone --single-branch --branch ${MIGRAPHX_BRANCH} --recursive https://github.com/ROCm/AMDMIGraphX.git \
    && cd AMDMIGraphX \
    && rbuild build -d depend -DBUILD_TESTING=Off -DCMAKE_INSTALL_PREFIX=/opt/rocm/ --cxx=/opt/rocm/llvm/bin/clang++ -DGPU_TARGETS=${GPU_ARCH} \
    && cd build && make install

# Install torch_migraphx
RUN pip3 install pybind11-global
RUN cd /workspace/torch_migraphx/py && \
    export TORCH_CMAKE_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)") && \
    python -m pip install .

WORKDIR /workspace
ENV LD_LIBRARY_PATH=${ROCM_PATH}/lib
ENV PYTHONPATH=${ROCM_PATH}/lib