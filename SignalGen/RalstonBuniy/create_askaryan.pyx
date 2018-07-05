import numpy as np
cimport numpy as np

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

cdef extern from "createAsk.h":
    void getFrequencySpectrum2(double *& , double *& , double *& , double *& , double *& , double *& , int & ,
                               double, double, double * , int, int, double, double)

cpdef get_frequency_spectrum(energy, theta, freqs, is_em_shower, n, R):
    cdef:
        double * spectrumRealR
        double * spectrumImagR
        double * spectrumRealTheta
        double * spectrumImagTheta
        double * spectrumRealPhi
        double * spectrumImagPhi
        int size
        np.npy_intp shape[1]
        np.ndarray[double, mode = "c"] freqs2 = freqs

    getFrequencySpectrum2(spectrumRealR, spectrumImagR,
                          spectrumRealTheta, spectrumImagTheta,
                          spectrumRealPhi, spectrumImagPhi,
                          size,
                          energy, theta, & freqs2[0], len(freqs2), is_em_shower, n, R)

    # 1. Make sure that you have called np.import_array()
    # http://gael-varoquaux.info/programming/
    # cython-example-of-exposing-c-computed-arrays-in-python-without-data-copies.html
    # 2. OWNDATA flag is important. It tells the NumPy to free data when the python object is deleted.
    # https://stackoverflow.com/questions/23872946/force-numpy-ndarray-to-take-ownership-of-its-memory-in-cython/
    # You can verify that the memory gets freed when Python object is deleted by using tools such as pmap.
    shape[0] = < np.npy_intp > size
    cdef np.ndarray[double, ndim = 1] spec_realx = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, spectrumRealR)
    PyArray_ENABLEFLAGS(spec_realx, np.NPY_OWNDATA)
    cdef np.ndarray[double, ndim = 1] spec_imagx = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, spectrumImagR)
    PyArray_ENABLEFLAGS(spec_imagx, np.NPY_OWNDATA)
    cdef np.ndarray[double, ndim = 1] spec_realy = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, spectrumRealTheta)
    PyArray_ENABLEFLAGS(spec_realy, np.NPY_OWNDATA)
    cdef np.ndarray[double, ndim = 1] spec_imagy = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, spectrumImagTheta)
    PyArray_ENABLEFLAGS(spec_imagy, np.NPY_OWNDATA)
    cdef np.ndarray[double, ndim = 1] spec_realz = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, spectrumRealPhi)
    PyArray_ENABLEFLAGS(spec_realz, np.NPY_OWNDATA)
    cdef np.ndarray[double, ndim = 1] spec_imagz = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, spectrumImagPhi)
    PyArray_ENABLEFLAGS(spec_imagz, np.NPY_OWNDATA)
    eR = spec_realx + 1j * spec_imagx
    eTheta = spec_realy + 1j * spec_imagy
    ePhi = spec_realz + 1j * spec_imagz

    return eR, eTheta, ePhi


