ARG DOCKER_TAG
FROM ${DOCKER_TAG}

RUN yum install -y python3-devel migraphx-devel
RUN pip3 install pybind11-global
