# Contributing to Torch-MIGraphX

Thank you for your interest in contributing to Torch-MIGraphX! This guide covers everything you need to get started.

For general ROCm contribution guidelines, see the [ROCm contributing guide](https://github.com/ROCm/ROCm/blob/develop/CONTRIBUTING.md).

## Development Setup

### Prerequisites

- [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/) (with MIGraphX installed)
- [PyTorch (ROCm version)](https://rocm.docs.amd.com/projects/install-on-linux/en/develop/how-to/3rd-party/pytorch-install.html#using-a-wheels-package)
- Python 3.7+
- A C++ compiler compatible with your PyTorch installation (for building the native extension)

### Using Docker (Recommended)

The easiest way to get a working development environment is with the provided dev container.

Run these from the **repo root** so the whole repo is mounted into the container at `/workspace/torch_migraphx`:

```bash
docker build -f docker/dev.Dockerfile -t torch_migraphx_dev .
docker run -it --network=host --device=/dev/kfd --device=/dev/dri \
    --group-add=video --ipc=host \
    -v "$(pwd)":/workspace/torch_migraphx \
    torch_migraphx_dev
```

### Local Setup

```bash
git clone https://github.com/ROCm/torch_migraphx.git
cd torch_migraphx/py
pip install -e .
```

The C++ extension is compiled ahead-of-time if PyTorch is available in the build environment. Otherwise, it will be compiled just-in-time on first import.

### Running Tests

Tests require a system with a ROCm-compatible GPU, MIGraphX, and PyTorch (ROCm) installed.

```bash
cd tests
pytest fx/ -v
pytest dynamo/ -v
```

## Branch Strategy

- The default integration branch is **`master`**.
- All pull requests should target **`master`** unless otherwise discussed with maintainers.
- Feature and fix branches are created from `master`.
- Use descriptive branch names: `fix/short-description`, `feature/short-description`, or `users/<github-username>/description`.

## Coding Standards

### Python

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions.
- Use type hints for public API functions.
- Keep imports organized: standard library, third-party, then local.

### General

- Write clear, self-documenting code. Avoid comments that merely restate what the code does.
- Every new feature or bug fix should include corresponding tests.
- Commit messages should use the imperative mood (e.g., "Fix shape mismatch in converter" not "Fixed shape mismatch").

## Pull Request Process

1. **Search existing issues** before starting work to avoid duplicates. If no issue exists, consider opening one first to discuss the approach.
2. **Create a branch** from `master`.
3. **Make your changes** with appropriate test coverage.
4. **Ensure tests pass** locally before submitting.
5. **Open a pull request** targeting `master` with a clear description of:
   - What the change does and why
   - How it was tested
   - Any breaking changes or migration steps
6. **Address review feedback** — maintainers may request changes before merging.
7. **CI must pass** — all automated checks need to be green before merge.

### PR Checklist

- [ ] Code builds and tests pass locally
- [ ] New/changed functionality has test coverage
- [ ] Commit messages are clear and use imperative mood
- [ ] Documentation updated if applicable (README, docstrings, examples)

## Reporting Issues

- Use [GitHub Issues](https://github.com/ROCm/torch_migraphx/issues) for bugs and feature requests.
- Include ROCm version, PyTorch version, GPU model, and steps to reproduce when filing bugs.
- For security vulnerabilities, see [SECURITY.md](SECURITY.md).

## License

By contributing, you agree that your contributions will be licensed under the [BSD 3-Clause License](LICENSE) that covers this project.
