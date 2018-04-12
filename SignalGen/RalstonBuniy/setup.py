from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy


extensions = [
    Extension('createAsk_call', ['createAsk_call.pyx', 'createAsk.cpp'],
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