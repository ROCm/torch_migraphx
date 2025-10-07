# Example use:
# docker build -t tm_build -f ci/wheel/build.Dockerfile .
# docker run -it -v$(pwd):/workspace/torch_migraphx tm_build /bin/bash /workspace/torch_migraphx/ci/wheel/build_whl.sh

export PROJECT_DIR=/workspace/torch_migraphx

# PY_BUILD_CODE, PY_VERSION
build_audit_whl() {
    /opt/python/$1/bin/python -m pip install torch==2.7.1 torchvision==0.22.1 -f https://compute-artifactory.amd.com/artifactory/compute-pytorch-rocm/compute-rocm-dkms-no-npi-hipclang/16643/
    TORCH_LIB_DIR=/opt/python/$1/lib/python$2/site-packages/torch/lib/

    /opt/python/$1/bin/python setup.py clean bdist_wheel

    /opt/python/$1/bin/python -m pip install auditwheel
    LD_LIBRARY_PATH=${TORCH_LIB_DIR}:${LD_LIBRARY_PATH} /opt/python/$1/bin/python -m auditwheel repair $(cat ${PROJECT_DIR}/ci/wheel/excludes.params) --plat manylinux_2_28_x86_64 dist/torch_migraphx-*-$1-linux_x86_64.whl
}

build_py39(){
    cd ${PROJECT_DIR}/py
    build_audit_whl "cp39-cp39" "3.9"
}

build_py310(){
    cd ${PROJECT_DIR}/py
    build_audit_whl "cp310-cp310" "3.10"
}

build_py311(){
    cd ${PROJECT_DIR}/py
    build_audit_whl "cp311-cp311" "3.11"
}

build_py312(){
    cd ${PROJECT_DIR}/py
    build_audit_whl "cp312-cp312" "3.12"
}

echo "Python version is: $PYTHON_VERSION"
if [[ "$PYTHON_VERSION" == *"3.9"* ]]; then
    build_py39
fi
if [[ "$PYTHON_VERSION" == *"3.10"* ]]; then
    build_py310
fi
if [[ "$PYTHON_VERSION" == *"3.11"* ]]; then
    build_py311
fi
if [[ "$PYTHON_VERSION" == *"3.12"* ]]; then
    build_py312
fi