from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy


extensions = [
    Extension('create_askaryan', ['create_askaryan.pyx', 'createAsk.cpp', 'Askaryan.cxx'],
              include_dirs=[numpy.get_include(), '../../utilities/'],
#               extra_compile_args=['-std=c++11'],
              libraries=['fftw3'],
              language='c++'
              ),
]

setup(
    ext_modules=cythonize(extensions),
    cmdclass = {'build_ext': build_ext}
#     extra_compile_args=["-w", '-g']
)