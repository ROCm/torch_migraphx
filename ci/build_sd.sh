# Example use:
# docker build -t tm_build -f build.Dockerfile .
# docker run -it -v$(pwd):/workspace/torch_migraphx tm_build /bin/bash /workspace/torch_migraphx/ci/build_sd.sh


# PY_BUILD_CODE=cp38-cp38
# PY_VERSION=3.8
# PY_NAME=python${PY_VERSION}
# PY_DIR=/opt/python/${PY_BUILD_CODE}
# PY_PKG_DIR=${PY_DIR}/lib/${PY_NAME}/site-packages/

# ${PY_DIR}/bin/python -m pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/rocm5.6
# ${PY_DIR}/bin/python setup.py bdist_wheel

# ${PY_DIR}/bin/python -m pip install auditwheel
# ${PY_DIR}/bin/python -m auditwheel repair  $(cat ${PROJECT_DIR}/ci/excludes.params) --plat manylinux_2_17_x86_64 dist/torch_migraphx-*-${PY_BUILD_CODE}-linux_x86_64.whl


export PROJECT_DIR=/workspace/torch_migraphx
cd /workspace/torch_migraphx/py
PY_BUILD_CODE=cp38-cp38
PY_DIR=/opt/python/${PY_BUILD_CODE}
${PY_DIR}/bin/python -m pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm5.7
${PY_DIR}/bin/python -m pip install -r requirements.txt
${PY_DIR}/bin/python setup.py clean sdist

#  ${PY_DIR}/bin/python -m auditwheel repair  $(cat ${PROJECT_DIR}/ci/soname_excludes.params) --plat manylinux_2_17_x86_64 dist/torch_migraphx-*-${PY_BUILD_CODE}-linux_x86_64.whl