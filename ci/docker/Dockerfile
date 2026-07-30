ARG PYTORCH_IMAGE=rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.11.0
FROM ${PYTORCH_IMAGE}
ARG ROCM_PATH=/opt/rocm
ARG ROCM_SDK_ROOT=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel
ARG ROCM_CMAKE_PREFIX=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/cmake
ARG ROCM_WHEEL_INDEX=https://repo.amd.com/rocm/whl-multi-arch/
ARG C_COMPILER=/opt/venv/bin/amdclang
ARG CXX_COMPILER=/opt/venv/bin/amdclang++
ARG MIGRAPHX_BRANCH
ENV MIGRAPHX_BRANCH=${MIGRAPHX_BRANCH:-"rocm-7.14"}
ARG GPU_ARCH
ENV GPU_ARCH=${GPU_ARCH:-"gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1201"}
ARG WHL_URL

# Install Dependencies: MIGraphX
RUN apt-get update && apt-get install -y --no-install-recommends cmake && rm -rf /var/lib/apt/lists/*
RUN if command -v rocm-sdk >/dev/null 2>&1; then \
        pip install --index-url ${ROCM_WHEEL_INDEX} "rocm[devel]==$(rocm-sdk version)" \
        && rocm-sdk init; \
    fi
# Install rbuild
RUN pip install https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz
# install migraphx from source
RUN git clone --single-branch --branch ${MIGRAPHX_BRANCH} --recursive https://github.com/ROCm/AMDMIGraphX.git \
    && cd AMDMIGraphX \
    && if [ -d "${ROCM_CMAKE_PREFIX}" ]; then CMAKE_SEARCH_PATH="${ROCM_CMAKE_PREFIX}"; else CMAKE_SEARCH_PATH="${ROCM_PATH}"; fi \
    && if [ -d "${ROCM_SDK_ROOT}" ]; then CXX_FLAGS="--rocm-path=${ROCM_SDK_ROOT}"; else CXX_FLAGS=""; fi \
    && CMAKE_PREFIX_PATH="${CMAKE_SEARCH_PATH}" rbuild build -d depend -DBUILD_TESTING=Off -DMIGRAPHX_ENABLE_GPU=On -DCMAKE_INSTALL_PREFIX=${ROCM_PATH} -DCMAKE_PREFIX_PATH="${CMAKE_SEARCH_PATH}" -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" --cc=${C_COMPILER} --cxx=${CXX_COMPILER} -DGPU_TARGETS=${GPU_ARCH} \
    && cd build \
    && make install


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
