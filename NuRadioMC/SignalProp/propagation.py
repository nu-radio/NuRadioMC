from __future__ import absolute_import, division, print_function

solution_types = {1: 'direct',
                      2: 'refracted',
                      3: 'reflected'}
solution_types_revert = {v:k for k, v in solution_types.items()}

available_modules = ['analytic',
                     'radiopropa',
                     'direct_ray']

reflection_case = {1: 'upwards launch vector',
                   2: 'downward launch vector'}

def get_propagation_module(name=None):
    """
    wrapper around all propagation modules
    
    The function returns the python class of the 
    respective propagation module
    
    Parameters
    ----------
    name: string
        Which ray tracing module to use. Options are:

        * "analytic" : analytic ray tracer. Requires that the index of refraction
          is of an exponential form.
        * "radiopropa" : the RadioPropa numerical ray tracer. Supports an arbitrary 
          index of refraction, but requires that RadioPropa is installed.
        * "direct_ray" : a dummy ray tracer that draws straight lines and 
          ignores refraction. Useful for debugging.

    """
    if name is None:
        from NuRadioMC.SignalProp.propagation_base_class import ray_tracing_base
        return ray_tracing_base
    elif(name==available_modules[0]):
        from NuRadioMC.SignalProp.analyticraytracing import ray_tracing
        return ray_tracing
    elif(name==available_modules[2]):
        from NuRadioMC.SignalProp.directRayTracing import direct_ray_tracing
        return direct_ray_tracing
    elif(name==available_modules[1]):
        from NuRadioMC.SignalProp.radioproparaytracing import radiopropa_ray_tracing
        return radiopropa_ray_tracing
        
    else:
        msg = "Module \'{}\' not implemented. Available modules: {}".format(
            name, str(available_modules))
        raise NotImplementedError(msg)
