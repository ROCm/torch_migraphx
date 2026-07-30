MIGRAPHX_BRANCH=${1:-"rocm-7.14"}
MIGRAPHX_REPO=https://github.com/ROCm/AMDMIGraphX.git
GPU_ARCH=${2:-"gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1201"}
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
ROCM_WHEEL_INDEX=${ROCM_WHEEL_INDEX:-https://repo.amd.com/rocm/whl-multi-arch/}
if [ -x /opt/rocm/llvm/bin/clang++ ]; then
    C_COMPILER=${C_COMPILER:-/opt/rocm/llvm/bin/clang}
    CXX_COMPILER=${CXX_COMPILER:-/opt/rocm/llvm/bin/clang++}
    ROCM_CMAKE_PREFIX=${ROCM_CMAKE_PREFIX:-/opt/rocm}
    CXX_FLAGS=${CXX_FLAGS:-}
else
    C_COMPILER=${C_COMPILER:-$(command -v amdclang)}
    CXX_COMPILER=${CXX_COMPILER:-$(command -v amdclang++)}
    pip3 install --index-url "$ROCM_WHEEL_INDEX" "rocm[devel]==$(rocm-sdk version)"
    rocm-sdk init
    ROCM_SDK_ROOT=${ROCM_SDK_ROOT:-$(rocm-sdk path --root)}
    ROCM_CMAKE_PREFIX=${ROCM_CMAKE_PREFIX:-$(rocm-sdk path --cmake)}
    CXX_FLAGS=${CXX_FLAGS:-"--rocm-path=${ROCM_SDK_ROOT}"}
fi

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

CMAKE_PREFIX_PATH="$ROCM_CMAKE_PREFIX" rbuild build -d depend -DBUILD_TESTING=Off -DMIGRAPHX_ENABLE_GPU=On -DCMAKE_INSTALL_PREFIX="$ROCM_PATH" -DCMAKE_PREFIX_PATH="$ROCM_CMAKE_PREFIX" -DCMAKE_CXX_FLAGS="$CXX_FLAGS" --cc="$C_COMPILER" --cxx="$CXX_COMPILER" -DGPU_TARGETS=$GPU_ARCH

cd build
make install
