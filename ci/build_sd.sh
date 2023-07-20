# Example use:
# docker build -t tm_build -f build.Dockerfile .
# docker run -it -v$(pwd):/workspace/torch_migraphx tm_build /bin/bash /workspace/torch_migraphx/ci/build_sd.sh

export PROJECT_DIR=/workspace/torch_migraphx
cd /workspace/torch_migraphx/py
PY_BUILD_CODE=cp38-cp38
PY_DIR=/opt/python/${PY_BUILD_CODE}
${PY_DIR}/bin/python -m pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm5.5
${PY_DIR}/bin/python -m pip install -r requirements.txt
${PY_DIR}/bin/python setup.py sdist
