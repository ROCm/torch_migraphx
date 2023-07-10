cd /workspace/torch_migraphx/py

# Example use:
# docker build -t tm_ci -f ci.Dockerfile .
# docker run -it -v$(pwd):/workspace/torch_migraphx tm_ci /bin/bash /workspace/torch_migraphx/build_whl.sh

build_py38() {
    /opt/python/cp38-cp38/bin/python -m pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm5.5
    /opt/python/cp38-cp38/bin/python -m pip install -r requirements.txt
    /opt/python/cp38-cp38/bin/python setup.py bdist_wheel
}

build_py39() {
    /opt/python/cp39-cp39/bin/python -m pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm5.5
    /opt/python/cp39-cp39/bin/python -m pip install -r requirements.txt
    /opt/python/cp39-cp39/bin/python setup.py bdist_wheel
}

build_py310() {
    /opt/python/cp310-cp310/bin/python -m pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm5.5
    /opt/python/cp310-cp310/bin/python -m pip install -r requirements.txt
    /opt/python/cp310-cp310/bin/python setup.py bdist_wheel
}

build_py311() {
    /opt/python/cp311-cp311/bin/python -m pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm5.5
    /opt/python/cp311-cp311/bin/python -m pip install -r requirements.txt
    /opt/python/cp311-cp311/bin/python setup.py bdist_wheel
}


build_py38
build_py39
build_py310
build_py311