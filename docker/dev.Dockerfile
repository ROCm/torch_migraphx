FROM rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.11.0

ARG ROCM_PATH=/opt/rocm
ARG ROCM_CMAKE_PREFIX=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/cmake
ARG ROCM_WHEEL_INDEX=https://repo.amd.com/rocm/whl-multi-arch/
ARG C_COMPILER=/opt/venv/bin/amdclang
ARG CXX_COMPILER=/opt/venv/bin/amdclang++
ARG MIGRAPHX_BRANCH=develop 
ARG GPU_ARCH="gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1201"

# Install Dependencies: MIGraphX
RUN apt-get update && apt-get install -y --no-install-recommends cmake && rm -rf /var/lib/apt/lists/*
RUN pip install --index-url ${ROCM_WHEEL_INDEX} "rocm[devel]==$(rocm-sdk version)" \
    && rocm-sdk init
# Install rbuild
RUN pip install https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz
# install migraphx from source
RUN git clone --single-branch --branch ${MIGRAPHX_BRANCH} --recursive https://github.com/ROCm/AMDMIGraphX.git \
    && cd AMDMIGraphX \
    && CMAKE_PREFIX_PATH="${ROCM_CMAKE_PREFIX}" rbuild build -d depend -DBUILD_TESTING=Off -DCMAKE_INSTALL_PREFIX=${ROCM_PATH} -DCMAKE_PREFIX_PATH="${ROCM_CMAKE_PREFIX}" --cc=${C_COMPILER} --cxx=${CXX_COMPILER} -DGPU_TARGETS=${GPU_ARCH} \
    && cd build && make install


# Install Dependencies: pybind-global
RUN pip3 install pybind11-global

ENV LD_LIBRARY_PATH=${ROCM_PATH}/lib
ENV PYTHONPATH=${ROCM_PATH}/lib