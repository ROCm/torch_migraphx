FROM pytorch/manylinux2_28-builder:rocm6.4

RUN yum install -y python3-devel migraphx-devel
RUN pip3 install pybind11-global
