FROM pytorch/manylinux-builder:rocm5.5

ARG ROCM_PATH=/opt/rocm

RUN yum install -y python3-devel
RUN pip3 install https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz
RUN pip3 install pybind11-global

COPY ./tools/install_migraphx.sh /
RUN /install_migraphx.sh && rm /install_migraphx.sh

# ENV PYTHONPATH=${PREFIX}
# ENV TORCH_USE_RTLD_GLOBAL=YES
# ENV LD_LIBRARY_PATH=${PREFIX}
ENV PYTHONPATH=${ROCM_PATH}/lib
ENV TORCH_USE_RTLD_GLOBAL=YES
ENV LD_LIBRARY_PATH=${ROCM_PATH}/lib