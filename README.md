<!-- START OF README TEMPLATE -->
<!-- Badges — replace YOUR-ORG and YOUR-REPO with actual values -->
[![License](https://img.shields.io/github/license/ROCm/torch_migraphx.svg?style=flat)](LICENSE)
[![Contributors](https://img.shields.io/github/contributors/ROCm/torch_migraphx.svg?style=flat)](https://github.com/ROCm/torch_migraphx/graphs/contributors)
<!-- Uncomment when CI is configured: -->
<!-- [![Build Status](https://github.com/ROCm/torch_migraphx/actions/workflows/ci.yml/badge.svg)](https://github.com/ROCm/torch_migraphx/actions) -->
<!-- [![OpenSSF Best Practices](https://www.bestpractices.dev/projects/518908750/badge)](https://www.bestpractices.dev/projects/518908750) -->
# Torch-MIGraphX
> Torch-MIGraphX integrates AMD's graph inference engine with the PyTorch ecosystem.

It provides utilities and APIs for generating a `mgx_module` that is designed to be invoked in the same manner as any other torch module, but utilize the MIGraphX inference engine internally. 

This library currently supports two paths for lowering:
1. FX Tracing: Uses tracing API provided by the `torch.fx` library.
2. Dynamo Backend: Importing torch_migraphx automatically registers the "migraphx" backend that can be used with the `torch.compile` API.

## Getting Started
### Docker
The simplest and recommended way to get started is using the provided Dockerfile.
Build using:
```
./build_image.sh
```
Start container using:
```
sudo docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined torch_migraphx
```

The default Dockerfile builds on the nightly pytorch container and installs the latest source version of MIGraphX and torch_migraphx. For more builds refer to the docker directory.

### Install From Source
Install Pre-reqs:
- [PyTorch (ROCM version)](https://rocm.docs.amd.com/projects/install-on-linux/en/develop/how-to/3rd-party/pytorch-install.html#using-a-wheels-package)
- [MIGraphX](https://github.com/ROCm/AMDMIGraphX?tab=readme-ov-file#installing-from-binaries)

Build and install from source
```
git clone https://github.com/ROCmSoftwarePlatform/torch_migraphx.git
cd ./torch_migraphx/py
pip install .
```

## Example Usage
```
# FX Tracing
torch_migraphx.fx.lower_to_mgx(torch_model, sample_inputs)

# Dynamo Backend
torch.compile(torch_model, backend="migraphx")
```

### Lower resnet50 using FX Tracing
```
import torch
import torchvision
import torch_migraphx

resnet = torchvision.models.resnet50()
sample_input = torch.randn(2, 3, 64, 64)
resnet_mgx = torch_migraphx.fx.lower_to_mgx(resnet, [sample_input])
result = resnet_mgx(sample_input)
```

### Lower densenet using torch.compile
```
import torch
import torchvision
import torch_migraphx

densenet = torchvision.models.densenet161().cuda()
sample_input = torch.randn(2, 3, 512, 512).cuda()
densenet_mgx = torch.compile(densenet, backend="migraphx")
result = densenet_mgx(sample_input.cuda())
```

For more examples please refer to the examples directory.


## Contributing
We welcome contributions! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, branch strategy, coding standards, and the pull request process.
For bugs and feature requests, open a [GitHub Issue](../../issues).
---
## Security
To report a security vulnerability, **do not open a public GitHub issue**.
See [SECURITY.md](SECURITY.md) for our responsible disclosure policy.
---
## Contact
For questions, issues, or contributions, please reach out to the maintainers:
- Shivad Bhavsar — [@shivadbhavsar](https://github.com/shivadbhavsar) · Shivad.Bhavsar@amd.com

> Note: For internal or private AMD repositories, maintainers must list their AMD email address.
See [CODEOWNERS](.github/CODEOWNERS) for the full ownership list.
---
## License
This project is licensed under the [BSD 3-Clause License](LICENSE).