ARG BUILD_NUM
FROM rocm/pytorch-private:manylinux-hipclang-${BUILD_NUM}

RUN yum install -y python3-devel migraphx-devel
RUN pip3 install pybind11-global