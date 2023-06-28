MIGRAPHX_BRANCH=master
MIGRAPHX_REPO=https://github.com/ROCmSoftwarePlatform/AMDMIGraphX.git

git clone --single-branch --branch $MIGRAPHX_BRANCH --recursive $MIGRAPHX_REPO
cd AMDMIGraphX

rbuild build -d depend -DBUILD_DEV=On --cxx=/opt/rocm/llvm/bin/clang++ 

cd build
make install
