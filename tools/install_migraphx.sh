MIGRAPHX_BRANCH=${1:-"develop"}
MIGRAPHX_REPO=https://github.com/ROCm/AMDMIGraphX.git
GPU_ARCH=${2:-"gfx900;gfx906;gfx908;gfx90a;gfx1030;gfx1100;gfx1101;gfx1102;gfx940;gfx941;gfx942"}

# Install rbuild
pip3 install https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz

# Update rocm-cmake to required version for migraphx
git clone https://github.com/RadeonOpenCompute/rocm-cmake.git  
cd rocm-cmake 
git checkout 5a34e72d9f113eb5d028e740c2def1f944619595 
mkdir build 
cd build
cmake .. 
cmake --build . --target install
cd ../..

git clone --single-branch --branch $MIGRAPHX_BRANCH --recursive $MIGRAPHX_REPO
cd AMDMIGraphX

rbuild build -d depend -DBUILD_TESTING=Off -DCMAKE_INSTALL_PREFIX=/opt/rocm/ --cxx=/opt/rocm/llvm/bin/clang++ -DGPU_TARGETS=$GPU_ARCH

cd build
make install
