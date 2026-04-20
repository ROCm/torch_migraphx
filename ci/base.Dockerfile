FROM rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1

ARG ROCM_PATH=/opt/rocm
ARG MIGRAPHX_BRANCH="rocm-7.2.1" 
ARG GPU_ARCH="gfx908;gfx90a;gfx940;gfx941;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1201"

# Install Dependencies: MIGraphX
RUN apt-get update && apt-get install -y --no-install-recommends cmake && rm -rf /var/lib/apt/lists/*
# Install rbuild
RUN pip install https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz
# install migraphx from source
RUN git clone --single-branch --branch ${MIGRAPHX_BRANCH} --recursive https://github.com/ROCm/AMDMIGraphX.git \
    && cd AMDMIGraphX \
    && rbuild build -d depend -DBUILD_TESTING=Off -DCMAKE_INSTALL_PREFIX=/opt/rocm/ --cxx=/opt/rocm/llvm/bin/clang++ -DGPU_TARGETS=${GPU_ARCH} \
    && cd build && make install


# Install Dependencies: pybind-global, transformers
RUN pip3 install pybind11-global transformers==4.41.2

ENV LD_LIBRARY_PATH=${ROCM_PATH}/lib
ENV PYTHONPATH=${ROCM_PATH}/lib