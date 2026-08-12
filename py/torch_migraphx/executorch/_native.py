"""Import-time builder for the native MIGraphX ExecuTorch backend."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import sys
import threading
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch

_BACKEND_NAME = "MIGraphXBackend"
_BUILD_LOCK = threading.Lock()
_LOADED_LIBRARY: Optional[str] = None


def _package_source_root() -> Path:
    package_root = Path(__file__).resolve().parent
    packaged = package_root / "csrc"
    if (packaged / "src" / "MIGraphXBackend.cpp").is_file():
        return packaged

    # Source-tree fallback used by editable installs.
    repository = package_root.parents[2]
    development = repository / "executorch" / "runtime"
    if (development / "src" / "MIGraphXBackend.cpp").is_file():
        return development

    raise RuntimeError(
        "Torch-MIGraphX ExecuTorch backend sources are missing. Reinstall "
        "torch_migraphx with its executorch/csrc package data."
    )


def _executorch_layout() -> Tuple[Path, Path, str]:
    try:
        import executorch
        from executorch.extension.pybindings import _portable_lib
    except ImportError as error:
        raise RuntimeError(
            "The native backend requires the ExecuTorch Python package"
        ) from error

    for root_value in executorch.__path__:
        root = Path(root_value)
        include = root / "include"
        pybindings = root / "extension" / "pybindings"
        libraries = sorted(pybindings.glob("_portable_lib*.so"))
        if include.is_dir() and libraries:
            try:
                version = importlib.metadata.version("executorch")
            except importlib.metadata.PackageNotFoundError:
                version = "unknown"
            # Retain the import: the plugin must register into this exact
            # portable runtime instance.
            del _portable_lib
            return include, libraries[0], version

    raise RuntimeError(
        "ExecuTorch is installed without native headers or _portable_lib"
    )


def _rocm_layout() -> Tuple[Path, Path, Path]:
    candidates = []
    configured = os.environ.get("ROCM_PATH")
    if configured:
        candidates.append(Path(configured))
    candidates.extend((Path("/opt/rocm"), Path("/usr")))

    for root in candidates:
        include = root / "include"
        lib = root / "lib"
        migraphx_library = lib / "libmigraphx_c.so"
        hip_library = lib / "libamdhip64.so"
        if (
            (include / "migraphx" / "migraphx.hpp").is_file()
            and (include / "hip" / "hip_runtime.h").is_file()
            and migraphx_library.is_file()
            and hip_library.is_file()
        ):
            return root, migraphx_library, hip_library

    raise RuntimeError(
        "Could not find ROCm, MIGraphX, and HIP development files. Set "
        "ROCM_PATH to the ROCm installation prefix."
    )


def _source_files(source_root: Path) -> Sequence[Path]:
    sources = (
        source_root / "src" / "MIGraphXBlob.cpp",
        source_root / "src" / "MIGraphXBackend.cpp",
    )
    missing = [str(source) for source in sources if not source.is_file()]
    if missing:
        raise RuntimeError(
            "Missing native backend source files: " + ", ".join(missing)
        )
    return sources


def _build_key(
    source_root: Path,
    sources: Sequence[Path],
    executorch_version: str,
    portable_library: Path,
    rocm_root: Path,
    native_libraries: Sequence[Path],
) -> str:
    digest = hashlib.sha256()
    for source in sorted(
        list(sources) + list((source_root / "include").rglob("*.h"))
    ):
        digest.update(str(source.relative_to(source_root)).encode())
        digest.update(source.read_bytes())
    for value in (
        torch.__version__,
        str(torch.version.hip),
        executorch_version,
        str(portable_library.resolve()),
        str(portable_library.stat().st_size),
        str(portable_library.stat().st_mtime_ns),
        str(rocm_root.resolve()),
        sys.implementation.cache_tag,
        os.environ.get("CXX", ""),
    ):
        digest.update(value.encode())
        digest.update(b"\0")
    for library in native_libraries:
        digest.update(str(library.resolve()).encode())
        digest.update(str(library.stat().st_size).encode())
        digest.update(str(library.stat().st_mtime_ns).encode())
    return digest.hexdigest()[:16]


def _cache_directory(build_key: str) -> Path:
    configured = os.environ.get("TORCH_MIGRAPHX_EXECUTORCH_CACHE_DIR")
    if configured:
        root = Path(configured).expanduser()
    else:
        cache_home = Path(
            os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
        )
        root = cache_home / "torch_migraphx" / "executorch"
    directory = root / build_key
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _registered_backend_names() -> Sequence[str]:
    from executorch.extension.pybindings import _portable_lib

    return tuple(_portable_lib._get_registered_backend_names())


def native_backend_loaded() -> bool:
    """Return whether the Python ExecuTorch runtime has MIGraphX registered."""

    try:
        return _BACKEND_NAME in _registered_backend_names()
    except ImportError:
        return False


def load_native_backend() -> Optional[str]:
    """Compile, cache, and load the native backend into ExecuTorch."""

    global _LOADED_LIBRARY

    if native_backend_loaded():
        return _LOADED_LIBRARY

    with _BUILD_LOCK:
        if native_backend_loaded():
            return _LOADED_LIBRARY

        source_root = _package_source_root()
        sources = _source_files(source_root)
        executorch_include, portable_library, executorch_version = (
            _executorch_layout()
        )
        rocm_root, migraphx_library, hip_library = _rocm_layout()

        backend_header = (
            executorch_include
            / "executorch"
            / "runtime"
            / "backend"
            / "interface.h"
        )
        compatibility_header = (
            source_root
            / "include"
            / "executorch"
            / "runtime"
            / "backend"
            / "interface.h"
        )
        if not backend_header.is_file() and not compatibility_header.is_file():
            raise RuntimeError(
                "ExecuTorch does not ship runtime/backend/interface.h and "
                "Torch-MIGraphX has no compatibility header for this release"
            )
        if not backend_header.is_file() and not executorch_version.startswith(
            "1.0."
        ):
            raise RuntimeError(
                "The bundled backend ABI compatibility header supports "
                f"ExecuTorch 1.0.x, not {executorch_version}"
            )

        include_paths = [source_root / "include", executorch_include]
        if backend_header.is_file():
            include_paths.reverse()
        include_paths.append(rocm_root / "include")

        build_key = _build_key(
            source_root,
            sources,
            executorch_version,
            portable_library,
            rocm_root,
            (migraphx_library, hip_library),
        )
        extension_name = f"_torch_migraphx_executorch_{build_key}"
        build_directory = _cache_directory(build_key)
        verbose = (
            os.environ.get(
                "TORCH_MIGRAPHX_EXECUTORCH_VERBOSE_BUILD", ""
            )
            == "1"
        )

        try:
            from torch.utils.cpp_extension import load

            library_path = load(
                name=extension_name,
                sources=[str(source) for source in sources],
                extra_include_paths=[str(path) for path in include_paths],
                extra_cflags=[
                    "-O2",
                    "-std=c++17",
                    "-DC10_USING_CUSTOM_GENERATED_MACROS",
                ],
                extra_ldflags=[
                    str(portable_library),
                    str(migraphx_library),
                    str(hip_library),
                    f"-Wl,-rpath,{portable_library.parent}",
                    f"-Wl,-rpath,{rocm_root / 'lib'}",
                    "-pthread",
                ],
                build_directory=str(build_directory),
                with_cuda=False,
                is_python_module=False,
                verbose=verbose,
            )
        except Exception as error:
            raise RuntimeError(
                "Failed to compile the Torch-MIGraphX ExecuTorch backend. "
                "Set TORCH_MIGRAPHX_EXECUTORCH_VERBOSE_BUILD=1 for the full "
                f"compiler command. Build cache: {build_directory}"
            ) from error

        if not native_backend_loaded():
            raise RuntimeError(
                "The native library loaded but MIGraphXBackend was not "
                "registered with ExecuTorch"
            )

        _LOADED_LIBRARY = str(library_path) if library_path else None
        return _LOADED_LIBRARY
