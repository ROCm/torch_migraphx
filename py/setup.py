#####################################################################################
# Copyright (c) 2022-present, Advanced Micro Devices, Inc. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#####################################################################################

import os
from setuptools import setup, find_packages

__version__ = open("version.txt").read().strip()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# When PyTorch is available at build time, AOT-compile the C++ extension
# into the wheel. Otherwise, ship the .cpp source as package data and
# JIT-compile at first import (see torch_migraphx/_C.py).
ext_modules = []
cmdclass = {}
try:
    from torch.utils.cpp_extension import CppExtension, BuildExtension
    ext_modules = [CppExtension('_torch_migraphx', ['./torch_migraphx/csrc/torch_migraphx_py.cpp'])]
    cmdclass = {'build_ext': BuildExtension}
except ModuleNotFoundError:
    pass
except ImportError as e:
    import warnings
    warnings.warn(
        f"PyTorch found but failed to import cpp_extension: {e}\n"
        "The C++ extension will not be AOT-compiled. "
        "It will be JIT-compiled at first import instead."
    )

setup(
    name='torch_migraphx',
    version=__version__,
    author='AMD',
    author_email='Shivad.Bhavsar@amd.com',
    url='https://github.com/ROCmSoftwarePlatform/torch_migraphx',
    description='Intergrate PyTorch with MIGraphX acceleration engine',
    long_description_content_type='text/markdown',
    long_description=long_description,
    install_requires=[
    "torch>=1.11.0",
    "numpy>=1.20.0,<2.0",
    "packaging",
    "tabulate",
    ],
    packages=find_packages(),
    package_dir={'torch_migraphx': 'torch_migraphx'},
    package_data={'torch_migraphx': ['csrc/*.cpp']},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    license="BSD",
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    include_package_data=True,
)
