# FROM pytorch/manylinux-builder:rocm5.5
FROM pytorch/manylinux-builder:rocm5.7

# ARG ROCM_PATH=/opt/rocm

RUN yum install -y python3-devel migraphx-devel
RUN pip3 install pybind11-global

# RUN pip3 install cmake==3.22.1
# RUN pip3 install https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz
# COPY ./tools/install_migraphx.sh /
# RUN /install_migraphx.sh && rm /install_migraphx.sh

# ENV PYTHONPATH=${ROCM_PATH}/lib
# ENV LD_LIBRARY_PATH=${ROCM_PATH}/lib