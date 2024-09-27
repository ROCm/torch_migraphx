FROM pytorch/manylinux-builder:rocm6.2

RUN yum install -y python3-devel migraphx-devel
RUN pip3 install pybind11-global
