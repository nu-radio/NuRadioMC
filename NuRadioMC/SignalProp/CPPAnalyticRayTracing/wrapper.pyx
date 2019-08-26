import numpy as np
cimport numpy as np
from operator import itemgetter
import time

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

cdef extern from "analytic_raytracing.cpp":
    void find_solutions2(double * &, double * &, int * &, int & , double, double, double, double, double, double, double, int, int, double)
    double get_attenuation_along_path2(double, double, double, double, double, double, double, double, double, int)
    

cpdef find_solutions(x1, x2, n_ice, delta_n, z_0, reflection, reflection_case, ice_reflection):
    cdef:
        double * C0s
        double * C1s
        int * types
        int size
        np.npy_intp shape[1]

#     t = time.time()
    find_solutions2(C0s, C1s, types, size, x1[0], x1[1], x2[0], x2[1], n_ice, delta_n, z_0, reflection, reflection_case, ice_reflection)
#     print((time.time() - t) * 1000)

    # 1. Make sure that you have called np.import_array()
    # http://gael-varoquaux.info/programming/
    # cython-example-of-exposing-c-computed-arrays-in-python-without-data-copies.html
    # 2. OWNDATA flag is important. It tells the NumPy to free data when the python object is deleted.
    # https://stackoverflow.com/questions/23872946/force-numpy-ndarray-to-take-ownership-of-its-memory-in-cython/
    # You can verify that the memory gets freed when Python object is deleted by using tools such as pmap.
    shape[0] = < np.npy_intp > size
    cdef np.ndarray[double, ndim = 1] C0ss = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, C0s)
    PyArray_ENABLEFLAGS(C0ss, np.NPY_OWNDATA)
    cdef np.ndarray[double, ndim = 1] C1ss = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, C1s)
    PyArray_ENABLEFLAGS(C1ss, np.NPY_OWNDATA)
    cdef np.ndarray[int, ndim = 1] typess = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT, types)
    PyArray_ENABLEFLAGS(typess, np.NPY_OWNDATA)

    solutions = []
    for i in range(len(C0ss)):
        solutions.append({'type': typess[i],
                          'C0': C0ss[i],
                          'C1': C1ss[i],
                          'reflection': reflection,
                          'reflection_case': reflection_case})

    s = sorted(solutions, key=itemgetter('type'))
#     print((time.time() - t) * 1000)
    return s


cpdef get_attenuation_along_path(x1, x2, C0, frequency, n_ice, delta_n, z_0, model):

#     t = time.time()
    return get_attenuation_along_path2(x1[0], x1[1], x2[0], x2[1], C0, frequency, n_ice, delta_n, z_0, model)
#     print((time.time() - t) * 1000)
