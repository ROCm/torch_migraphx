MIGRAPHX_BRANCH=master
MIGRAPHX_REPO=https://github.com/ROCmSoftwarePlatform/AMDMIGraphX.git
GPU_ARCH=${1:-"gfx900;gfx906;gfx908;gfx90a;gfx1030;gfx1100;gfx1101;gfx1102;gfx940;gfx941;gfx942"}

git clone --single-branch --branch $MIGRAPHX_BRANCH --recursive $MIGRAPHX_REPO
cd AMDMIGraphX

rbuild build -d depend -DCMAKE_INSTALL_PREFIX=/opt/rocm/ -DBUILD_DEV=On --cxx=/opt/rocm/llvm/bin/clang++ -DGPU_TARGETS=$GPU_ARCH

cd build
make install
