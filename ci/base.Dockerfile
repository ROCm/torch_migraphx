ARG PYTORCH_IMAGE=registry-sc-harbor.amd.com/framework/compute-rocm-dkms-no-npi-hipclang:16590_ubuntu24.04_py3.12_pytorch_release-2.8_db3ba667
FROM ${PYTORCH_IMAGE}
ARG ROCM_PATH=/opt/rocm
ARG MIGRAPHX_BRANCH
ENV MIGRAPHX_BRANCH=${MIGRAPHX_BRANCH:-"master"}
ARG GPU_ARCH
ENV GPU_ARCH=${GPU_ARCH:-"gfx900;gfx906;gfx908;gfx90a;gfx1030;gfx1100;gfx1101;gfx1102;gfx940;gfx941;gfx942"}
ARG ROCM_BUILD_JOB
ARG BUILD_NUM
ARG PYTHON_VERSION

# Install Dependencies: MIGraphX
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

# Install TorchMIGX wheel
RUN wget -r -nd -l1 -A "torch_migraphx*${PYTHON_VERSION}*manylinux*.whl " https://compute-artifactory.amd.com/artifactory/compute-pytorch-rocm/${ROCM_BUILD_JOB}/${BUILD_NUM}/ \
    && pip install --no-build-isolation torch_migraphx*${PYTHON_VERSION}*manylinux*.whl \
    && rm torch_migraphx*${PYTHON_VERSION}*manylinux*.whl 