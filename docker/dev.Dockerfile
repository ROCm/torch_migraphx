FROM rocm/pytorch-nightly:latest

ARG ROCM_PATH=/opt/rocm
ARG GPU_ARCH

# Install Dependencies: migraphx
COPY ./tools/install_migraphx.sh /
RUN /install_migraphx.sh develop ${GPU_ARCH} && rm /install_migraphx.sh

# Install Dependencies: pybind-global
RUN pip3 install pybind11-global

ENV LD_LIBRARY_PATH=${ROCM_PATH}/lib
ENV PYTHONPATH=${ROCM_PATH}/lib