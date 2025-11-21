from setuptools import setup, Extension
import pybind11
import sys

extra_args = ["-O3"]
if sys.platform == "win32":
    extra_args = ["/O2", "/arch:AVX2", "/fp:fast"]
else:

    extra_args = ["-O3", "-march=native", "-ffast-math"]

ext_modules = [
    Extension(
        "cpp_laplace",
        ["laplace.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=extra_args,
    ),
]

setup(
    name="cpp_laplace",
    version="0.1",
    ext_modules=ext_modules,
)