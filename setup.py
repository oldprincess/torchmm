from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension
import sys

cxx_flags = []
if sys.platform == "linux":
    cxx_flags = ["-std=c++17", "-O3"]
    nvcc_flags = ["-O3", "-std=c++17"]
else:
    cxx_flags = ["/std:c++17", "/O2", "/Zc:__cplusplus"]
    nvcc_flags = ["-O3", "-std=c++17", "-Xcompiler", "/Zc:__cplusplus"]

setup(
    name="torchmm",
    version="1.0.0",
    description="torch cuda matmul extension for integral",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="oldprincess",
    author_email="zirui.gong@foxmail.com",
    packages=find_packages(exclude=("test", "examples")),
    ext_modules=[
        cpp_extension.CUDAExtension(
            name="torchmm" + "._C",
            sources=[
                "torchmm/csrc/cuda/matmul.cu",
                "torchmm/csrc/torchmm.cpp",
            ],
            include_dirs=["third_party/cutlass/include"],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
        )
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
