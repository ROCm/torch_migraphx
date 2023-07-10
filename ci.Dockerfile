FROM pytorch/manylinux-builder:rocm5.5

ARG PREFIX=/usr/local

RUN yum install -y python3-devel
RUN pip3 install https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz
RUN pip3 install pybind11-global

COPY ./tools/install_migraphx.sh /
RUN /install_migraphx.sh && rm /install_migraphx.sh

ENV PYTHONPATH=${PREFIX}
ENV TORCH_USE_RTLD_GLOBAL=YES
ENV LD_LIBRARY_PATH=${PREFIX}