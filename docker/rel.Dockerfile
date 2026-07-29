## Reference release environment for PyTorch 2.11 and ROCm 7.14.
FROM rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.11.0

ARG ROCM_PATH=/opt/rocm
ARG MIGRAPHX_BRANCH=rocm-7.14
ARG GPU_ARCH="gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1201"

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends cmake git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz
RUN git clone --single-branch --branch ${MIGRAPHX_BRANCH} --recursive \
        https://github.com/ROCm/AMDMIGraphX.git \
    && cd AMDMIGraphX \
    && rbuild build -d depend -DBUILD_TESTING=Off \
        -DCMAKE_INSTALL_PREFIX=${ROCM_PATH} \
        --cxx=${ROCM_PATH}/llvm/bin/clang++ \
        -DGPU_TARGETS=${GPU_ARCH} \
    && cd build \
    && make install

RUN pip install pybind11-global "torchao>=0.17.0" torch-migraphx

ENV LD_LIBRARY_PATH=${ROCM_PATH}/lib
ENV PYTHONPATH=${ROCM_PATH}/lib

