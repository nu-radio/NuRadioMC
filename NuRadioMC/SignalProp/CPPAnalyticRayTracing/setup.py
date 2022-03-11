from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import os

try:
    print('Your $GSLDIR = ' + str(os.environ['GSLDIR']))
except:
    print('You have either not installed GSL or have not set the system variable $GSLDIR.\
           See NuRadioMC wiki for further GSL details. ')

if __name__ == "__main__":
    extensions = [
        Extension('wrapper', ['wrapper.pyx'],
                include_dirs=[numpy.get_include(), '../../utilities/', str(os.environ['GSLDIR']) + '/include/'],
                library_dirs=[str(os.environ['GSLDIR']) + '/lib/'],
                extra_compile_args=['-O3',"-mfpmath=sse"],
                libraries=['gsl', 'gslcblas'],
                language='c++'
                ),
    ]

    setup(
        ext_modules=cythonize(extensions),
        cmdclass = {'build_ext': build_ext}
    # extra_compile_args=["-mfpmath=sse"]
    )
