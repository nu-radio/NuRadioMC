from operator import itemgetter

cdef extern from "analytic_raytracing.cpp":
    void find_solutions2(double * &, double * &, int * &, int & , double, double, double, double, double, double, double, int, int, double)
    double get_attenuation_along_path2(double, double, double, double, double, double, double, double, double, int)
    double get_attenuation_length_wrapper(double, double, int)

cpdef find_solutions(x1, x2, n_ice, delta_n, z_0, reflection, reflection_case, ice_reflection):
    cdef: # These will give a warning for not being assigned during compilation, but this is fine (they are assigned in find_solutions2)
        double * C0s
        double * C1s
        int * types
        int size

    find_solutions2(C0s, C1s, types, size, x1[0], x1[1], x2[0], x2[1], n_ice, delta_n, z_0, reflection, reflection_case, ice_reflection)

    solutions = []
    for i in range(size):
        solutions.append({'type': types[i],
                          'C0': C0s[i],
                          'C1': C1s[i],
                          'reflection': reflection,
                          'reflection_case': reflection_case})

    s = sorted(solutions, key=itemgetter('reflection', 'C0'))
    return s


cpdef get_attenuation_along_path(x1, x2, C0, frequency, n_ice, delta_n, z_0, model):
    return get_attenuation_along_path2(x1[0], x1[1], x2[0], x2[1], C0, frequency, n_ice, delta_n, z_0, model)

cpdef get_attenuation_length(z, freq, model):
    return get_attenuation_length_wrapper(z, freq, model)