from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import os

print('Your $GSLDIR = ' + str(os.environ['GSLDIR']))
extensions = [
    Extension('wrapper', ['wrapper.pyx'],
              #include_dirs=[numpy.get_include(), '../../utilities/'],
              include_dirs=[numpy.get_include(), '../../utilities/', str(os.environ['GSLDIR']) + '/include/'],
              extra_compile_args=['-O3'],
              libraries=['gsl', 'gslcblas'],
              language='c++'
              ),
]

setup(
    ext_modules=cythonize(extensions),
    cmdclass = {'build_ext': build_ext}
#     extra_compile_args=["-w", '-g']
)
