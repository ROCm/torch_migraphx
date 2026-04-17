try:
    from _torch_migraphx import *
except ModuleNotFoundError:
    import os
    import torch
    import torch.utils.cpp_extension

    def _jit_load_extension():
        name = "_torch_migraphx"
        src = os.path.join(os.path.dirname(__file__), "csrc", "torch_migraphx_py.cpp")

        if not os.path.isfile(src):
            raise RuntimeError(
                f"torch_migraphx C++ source not found at {src}. "
                "The package may not have been installed correctly. "
                "Please reinstall with: pip install torch_migraphx"
            )

        verbose = os.environ.get("TORCH_MIGRAPHX_VERBOSE_BUILD", "") == "1"
        try:
            return torch.utils.cpp_extension.load(
                name=name,
                sources=[src],
                verbose=verbose,
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to JIT-compile the torch_migraphx C++ extension.\n"
                "\n"
                f"Error: {e}\n"
                "\n"
                "To debug, re-run with verbose build output:\n"
                "  TORCH_MIGRAPHX_VERBOSE_BUILD=1 python -c 'import torch_migraphx'\n"
                "\n"
                f"Source file:     {src}\n"
                f"PyTorch version: {torch.__version__}"
            ) from e

    _mod = _jit_load_extension()
    from_blob = _mod.from_blob
