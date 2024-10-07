ARG BASE_DOCKER=rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0
FROM $BASE_DOCKER

ARG PYTORCH_VERSION="2.3.0"
ENV PYTORCH_BRANCH="v${PYTORCH_VERSION}"
ENV WORKSPACE_DIR=/workspace
RUN sudo mkdir -p $WORKSPACE_DIR && sudo chown $(whoami):$(whoami) $WORKSPACE_DIR
WORKDIR $WORKSPACE_DIR

RUN sudo apt-get update -y && sudo apt-get install -y \
    cmake \
    python3 \
    python3-dev \
    python3-pip \
    software-properties-common && \
    sudo apt-get clean

# Install rbuild
RUN pip3 install https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz

# Install pip reqs
RUN pip3 install --upgrade pip
RUN pip3 install matplotlib==3.8.4 pandas==2.2.2 tabulate==0.9.0 gitpython pyyaml numpy==1.22.4 pybind11-global

# Installing torchaudio
RUN curl -o torchaudio.txt https://raw.githubusercontent.com/pytorch/pytorch/${PYTORCH_BRANCH}/.github/ci_commit_pins/audio.txt &&\
    torchaudio_hash="$(cat torchaudio.txt)" && \
    git clone https://github.com/pytorch/audio.git &&\
    cd audio &&\
    git checkout "$torchaudio_hash" &&\
    python3 setup.py install

# Installing torchdata
RUN curl -o torchdata.txt https://raw.githubusercontent.com/pytorch/pytorch/${PYTORCH_BRANCH}/.github/ci_commit_pins/data.txt &&\
  torchdata_hash="$(cat torchdata.txt)" &&\
  git clone https://github.com/pytorch/data.git &&\
  cd data &&\
  git checkout "$torchdata_hash" &&\
  python3 setup.py install

# Installing torchtext
RUN curl -o torchtext.txt https://raw.githubusercontent.com/pytorch/pytorch/${PYTORCH_BRANCH}/.github/ci_commit_pins/text.txt &&\
  torchtext_hash="$(cat torchtext.txt)" &&\
  git clone https://github.com/pytorch/text.git &&\
  cd text &&\
  git checkout "$torchtext_hash" &&\
  python3 setup.py install

# Inductor benchmark prerequisites
RUN pip install z3-solver

# Installing benchmarks
RUN curl -o torchbench.txt https://raw.githubusercontent.com/pytorch/pytorch/${PYTORCH_BRANCH}/.github/ci_commit_pins/torchbench.txt &&\
  torchbench_hash=$(cat torchbench.txt) &&\
  git clone https://github.com/pytorch/benchmark.git &&\
  cd benchmark &&\
  git checkout "$torchbench_hash" &&\
  python3 install.py

ENV PYTHONPATH=${WORKSPACE_DIR}/benchmark:/opt/rocm/lib:${PYTHONPATH}
ENV LD_LIBRARY_PATH=/opt/rocm/lib
ENV ROCM_PATH=/opt/rocm
