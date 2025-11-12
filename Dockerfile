FROM rocm/pytorch:rocm6.4.4_ubuntu24.04_py3.12_pytorch_release_2.7.1
ARG ROCM_PATH=/opt/rocm
ARG MIGRAPHX_BRANCH=master 
ARG GPU_ARCH="gfx908;gfx90a;gfx1030;gfx1100;gfx1101;gfx1102;gfx942;gfx950;gfx1201"

COPY . /workspace/torch_migraphx

# Install Dependencies: MIGraphX
# Install rbuild
RUN pip install https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz
# install migraphx from source
RUN git clone https://github.com/ROCm/AMDMIGraphX.git \
    && cd AMDMIGraphX && git checkout ${MIGRAPHX_BRANCH} \
    && rbuild build -d depend -DBUILD_TESTING=Off -DCMAKE_INSTALL_PREFIX=/opt/rocm/ --cxx=/opt/rocm/llvm/bin/clang++ -DGPU_TARGETS=${GPU_ARCH} \
    && cd build && make install

# Install torch_migraphx
RUN cd /workspace/torch_migraphx/py && python -m pip install . --no-build-isolation

WORKDIR /workspace
ENV LD_LIBRARY_PATH=${ROCM_PATH}/lib
ENV PYTHONPATH=${ROCM_PATH}/lib