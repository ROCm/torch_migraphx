# Example use:
# docker build -t tm_build -f ci/build.Dockerfile .
# docker run -it -v$(pwd):/workspace/torch_migraphx tm_build /bin/bash /workspace/torch_migraphx/ci/build_whl.sh

export PROJECT_DIR=/workspace/torch_migraphx

# PY_BUILD_CODE, PY_VERSION
build_audit_whl() {
    /opt/python/$1/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.6
    TORCH_LIB_DIR=/opt/python/$1/lib/python$2/site-packages/

    /opt/python/$1/bin/python setup.py clean bdist_wheel

    /opt/python/$1/bin/python -m pip install auditwheel
    LD_LIBRARY_PATH=${TORCH_LIB_DIR}:${LD_LIBRARY_PATH} /opt/python/$1/bin/python -m auditwheel repair  $(cat ${PROJECT_DIR}/ci/excludes.params) --plat manylinux_2_17_x86_64 dist/torch_migraphx-*-$1-linux_x86_64.whl
}

build_py37(){
    cd ${PROJECT_DIR}/py
    build_audit_whl "cp37-cp37m" "3.7"
}

build_py38(){
    cd ${PROJECT_DIR}/py
    build_audit_whl "cp38-cp38" "3.8"
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

# PyTorch wheels officially available for versions 3.8 - 3.11
build_py38
build_py39
build_py310
build_py311