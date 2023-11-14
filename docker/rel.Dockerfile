FROM ubuntu:20.04

# Install rocm key
RUN apt-get update && apt-get install -y gnupg2 --no-install-recommends curl && \
    curl -sL http://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -

# Add rocm repository
RUN sh -c 'echo deb [arch=amd64 trusted=yes] http://repo.radeon.com/rocm/apt/5.7/ focal main > /etc/apt/sources.list.d/rocm.list'

# From docs.amd.com for installing rocm. Needed to install properly
RUN sh -c "echo 'Package: *\nPin: release o=repo.radeon.com\nPin-priority: 600' > /etc/apt/preferences.d/rocm-pin-600"

# Install dependencies hipfft?
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    migraphx \
    python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch 2.1 stable releace for rocm
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
RUN pip3 install pybind11-global