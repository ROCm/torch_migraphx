MIGRAPHX_BRANCH=develop
MIGRAPHX_REPO=https://github.com/ROCmSoftwarePlatform/AMDMIGraphX.git

git clone --single-branch --branch $MIGRAPHX_BRANCH --recursive $MIGRAPHX_REPO
cd AMDMIGraphX

rbuild build -d depend --cxx=/opt/rocm/llvm/bin/clang++

cd build
make install
