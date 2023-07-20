# Example use:
# docker build -t tm_build -f build.Dockerfile .
# docker run -it -v$(pwd):/workspace/torch_migraphx tm_build /bin/bash /workspace/torch_migraphx/ci/build_whl.sh

export PROJECT_DIR=/workspace/torch_migraphx

build_py38() {
    cd /workspace/torch_migraphx/py
    PY_BUILD_CODE=cp38-cp38
    PY_DIR=/opt/python/${PY_BUILD_CODE}
    ${PY_DIR}/bin/python -m pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm5.5
    ${PY_DIR}/bin/python -m pip install -r requirements.txt
    ${PY_DIR}/bin/python -m pip install auditwheel
    ${PY_DIR}/bin/python setup.py bdist_wheel
    ${PY_DIR}/bin/python -m auditwheel repair $(cat ${PROJECT_DIR}/ci/soname_excludes.params) --plat manylinux2014_x86_64 dist/torch_migraphx-*-${PY_BUILD_CODE}-linux_x86_64.whl
}

build_py39() {
    cd /workspace/torch_migraphx/py
    PY_BUILD_CODE=cp39-cp39
    PY_DIR=/opt/python/${PY_BUILD_CODE}
    ${PY_DIR}/bin/python -m pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm5.5
    ${PY_DIR}/bin/python -m pip install -r requirements.txt
    ${PY_DIR}/bin/python -m pip install auditwheel
    ${PY_DIR}/bin/python setup.py bdist_wheel
    ${PY_DIR}/bin/python -m auditwheel repair $(cat ${PROJECT_DIR}/ci/soname_excludes.params) --plat manylinux2014_x86_64 dist/torch_migraphx-*-${PY_BUILD_CODE}-linux_x86_64.whl
}

build_py310() {
    cd /workspace/torch_migraphx/py
    PY_BUILD_CODE=cp310-cp310
    PY_DIR=/opt/python/${PY_BUILD_CODE}
    ${PY_DIR}/bin/python -m pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm5.5
    ${PY_DIR}/bin/python -m pip install -r requirements.txt
    ${PY_DIR}/bin/python -m pip install auditwheel
    ${PY_DIR}/bin/python setup.py bdist_wheel
    ${PY_DIR}/bin/python -m auditwheel repair $(cat ${PROJECT_DIR}/ci/soname_excludes.params) --plat manylinux2014_x86_64 dist/torch_migraphx-*-${PY_BUILD_CODE}-linux_x86_64.whl
}

build_py311() {
    cd /workspace/torch_migraphx/py
    PY_BUILD_CODE=cp311-cp311
    PY_DIR=/opt/python/${PY_BUILD_CODE}
    ${PY_DIR}/bin/python -m pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm5.5
    ${PY_DIR}/bin/python -m pip install -r requirements.txt
    ${PY_DIR}/bin/python -m pip install auditwheel
    ${PY_DIR}/bin/python setup.py bdist_wheel
    ${PY_DIR}/bin/python -m auditwheel repair $(cat ${PROJECT_DIR}/ci/soname_excludes.params) --plat manylinux2014_x86_64 dist/torch_migraphx-*-${PY_BUILD_CODE}-linux_x86_64.whl
}


build_py38
build_py39
build_py310
build_py311