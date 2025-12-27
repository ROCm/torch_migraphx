ARG JOB_NAME
ARG BUILD_NUM
FROM rocm/pytorch-private:manylinux-7.2-${JOB_NAME}-${BUILD_NUM}

RUN yum install -y python3-devel migraphx-devel
RUN pip3 install pybind11-global
