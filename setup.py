from pathlib import Path
from setuptools import setup, find_packages
from torch.utils import cpp_extension
import os
import sys


def _get_compile_flags(platform_name):
    if platform_name.startswith("linux"):
        return ["-std=c++17", "-O3", "-fopenmp"], ["-O3", "-std=c++17"]

    if platform_name.startswith("win") or platform_name in {"cygwin", "msys"}:
        return ["/std:c++17", "/O2", "/Zc:__cplusplus", "/openmp"], [
            "-O3",
            "-std=c++17",
            "-Xcompiler",
            "/Zc:__cplusplus",
        ]

    return ["-std=c++17", "-O3"], ["-O3", "-std=c++17"]


_build_extension_base = (
    cpp_extension.BuildExtension
    if isinstance(cpp_extension.BuildExtension, type)
    else object
)


class TorchmmBuildExtension(_build_extension_base):
    def _check_abi(self):
        if sys.platform.startswith("win"):
            return "msvc", "unknown"
        parent_check_abi = getattr(super(), "_check_abi", None)
        if parent_check_abi is None:
            return "unknown", "unknown"
        return parent_check_abi()


cxx_flags, nvcc_flags = _get_compile_flags(sys.platform)
force_cpu_only = os.environ.get("TORCHMM_FORCE_CPU_ONLY") == "1"
extension_cls = cpp_extension.CppExtension if force_cpu_only else cpp_extension.CUDAExtension
sources = [
    "torchmm/csrc/cpu/wide_binary_ops.cpp",
    "torchmm/csrc/cpu/wide_mul_ops.cpp",
    "torchmm/csrc/cpu/wide_shift_ops.cpp",
    "torchmm/csrc/cpu/wide_bmm_ops.cpp",
    "torchmm/csrc/wide/common.cpp",
    "torchmm/csrc/wide/dispatch.cpp",
    "torchmm/csrc/torchmm.cpp",
]
if not force_cpu_only:
    sources.insert(4, "torchmm/csrc/cuda/matmul.cu")
    sources.insert(5, "torchmm/csrc/cuda/wide_cutlass_probe.cu")
    sources.insert(6, "torchmm/csrc/cuda/wide_cuda_ops.cu")

define_macros = []
if not force_cpu_only:
    define_macros.append(("TORCHMM_WITH_CUDA", "1"))

extra_compile_args = {"cxx": cxx_flags}
if not force_cpu_only:
    extra_compile_args["nvcc"] = nvcc_flags

setup(
    name="torchmm",
    version="1.1.0",
    description="torch cuda matmul extension for integral",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="oldprincess",
    author_email="zirui.gong@foxmail.com",
    packages=find_packages(exclude=("test", "examples")),
    ext_modules=[
        extension_cls(
            name="torchmm._C",
            sources=sources,
            include_dirs=[
                "third_party/cutlass/include",
                "third_party/wideint/include",
            ],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    cmdclass={"build_ext": TorchmmBuildExtension},
)
