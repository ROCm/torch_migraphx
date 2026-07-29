ARG PYTORCH_IMAGE=rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.11.0
FROM ${PYTORCH_IMAGE}
ARG ROCM_PATH=/opt/rocm
ARG MIGRAPHX_BRANCH
ENV MIGRAPHX_BRANCH=${MIGRAPHX_BRANCH:-"rocm-7.14"}
ARG GPU_ARCH
ENV GPU_ARCH=${GPU_ARCH:-"gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1201"}
ARG WHL_URL

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

# PT2E moved out of PyTorch in 2.11. Keep older images on their in-tree API.
RUN if python3 -c 'import sys, torch; from packaging import version; sys.exit(0 if version.parse(torch.__version__) >= version.parse("2.11.dev") else 1)'; then \
        pip3 install "torchao>=0.17.0"; \
    fi

# Install TorchMIGX wheel
RUN wget -r -nd -l1 ${WHL_URL} \
    && pip3 install torch_migraphx*.whl \
    && rm torch_migraphx*.whl 
