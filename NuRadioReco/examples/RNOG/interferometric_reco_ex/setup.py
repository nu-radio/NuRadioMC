"""
Setup script for building the fast_delay_matrices C++ extension module

Build with: python setup.py build_ext --inplace
"""

from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'fast_delay_matrices',
        ['fast_delay_matrices.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        #extra_compile_args=['-std=c++17', '-O3', '-march=native', '-ffast-math'],
        # Use generic x86-64 target instead of -march=native for cluster compatibility
        # Slight performance cost (<10%) but works on all nodes without recompilation
        extra_compile_args=['-std=c++17', '-O3', '-march=x86-64', '-ffast-math'],
    ),
]

setup(
    name='fast_delay_matrices',
    version='1.0',
    author='Your Name',
    description='Fast C++ implementation of delay matrix computation',
    ext_modules=ext_modules,
    zip_safe=False,
)
