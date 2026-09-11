import os
import sys
import platform
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

if sys.platform == "darwin":
    machine = platform.machine().lower()
    if "arm" in machine:
        os.environ.setdefault("ARCHFLAGS", "-arch arm64")
    elif "x86" in machine:
        os.environ.setdefault("ARCHFLAGS", "-arch x86_64")


class BuildExt(build_ext):
    def build_extensions(self):
        compiler_type = self.compiler.compiler_type
        machine = platform.machine().lower()
        is_arm = machine in ("arm64", "aarch64") or ("arm" in machine)

        for ext in self.extensions:
            if compiler_type == "msvc":
                if is_arm:
                    ext.extra_compile_args = ["/O2", "/fp:fast"]
                else:
                    ext.extra_compile_args = ["/O2", "/arch:AVX2", "/fp:fast"]
            else:
                if is_arm:
                    ext.extra_compile_args = ["-O3", "-ffast-math", "-fPIC"]
                else:
                    ext.extra_compile_args = ["-O3", "-mavx2", "-mfma", "-ffast-math", "-fPIC"]
        super().build_extensions()


ext_modules = [
    Extension(
        "nanovector._ext",
        sources=[
            "src/nanovector_py.c",
            "src/nanovector.c",
            "src/nanovector_avx2.c",
            "src/nanovector_neon.c",
        ],
        include_dirs=["src"],
    )
]

setup(
    name="nanovector",
    version="0.1.3",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    install_requires=["numpy>=1.20"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    package_data={"nanovector": ["py.typed", "*.dll", "*.so", "*.dylib"]},
)
