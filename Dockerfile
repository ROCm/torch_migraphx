FROM rocm/pytorch:latest

ARG PREFIX=/usr/local

# Support multiarch
RUN dpkg --add-architecture i386

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    clang-format-10 \
    python3.7-dev \
    zip unzip &&\
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz

# Install Dependencies: migraphx
COPY ./tools/install_migraphx.sh /
RUN /install_migraphx.sh && rm /install_migraphx.sh

ENV PYTHONPATH=/opt/rocm/lib
ENV TORCH_USE_RTLD_GLOBAL=YES
ENV LD_LIBRARY_PATH=/opt/rocm/lib
