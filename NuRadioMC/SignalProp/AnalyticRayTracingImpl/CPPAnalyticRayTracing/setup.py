from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import os
import sys

if __name__ == "__main__":
    gsl_dir = os.environ.get('GSLDIR')
    if not gsl_dir:
        sys.exit('$GSLDIR is not set: set it to your GSL installation, e.g. export GSLDIR=$(gsl-config --prefix). See '
                 '"GSL speed-boost" in https://nu-radio.github.io/NuRadioMC/NuRadioMC/pages/Manuals/signal_propagation.html')
    print('Your $GSLDIR = ' + gsl_dir)

    extensions = [
        Extension('wrapper', ['wrapper.pyx'],
                include_dirs=[numpy.get_include(), '../../../utilities/', gsl_dir + '/include/'],
                library_dirs=[gsl_dir + '/lib/'],
                extra_compile_args=['-O3'],
                define_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
                libraries=['gsl', 'gslcblas'],
                language='c++'
                ),
    ]

    setup(
        ext_modules=cythonize(extensions, language_level=3),
        cmdclass = {'build_ext': build_ext}
    # extra_compile_args=["-mfpmath=sse"]
    )
