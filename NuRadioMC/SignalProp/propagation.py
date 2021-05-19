from __future__ import absolute_import, division, print_function

def get_propagation_module(name=None):
    """
    wrapper around all propagation modules
    
    The function returns the python class of the 
    respective propagation module
    
    Parameters
    ----------
    name: string
        * analytic: analytic ray tracer
    """
    if name is None:
        from NuRadioMC.SignalProp.propagation_base_class import ray_tracing_base
        return base_ray_tracing
    elif(name=='analytic'):
        from NuRadioMC.SignalProp.analyticraytracing import ray_tracing
        return analytic_ray_tracing
    elif(name=='direct_ray'):
        from NuRadioMC.SignalProp.directRayTracing import direct_ray_tracing
        return direct_ray_tracing
    elif(name=='radiopropa'):
        from NuRadioMC.SignalProp.radioproparaytracing import radiopropa_ray_tracing
        return radiopropa_ray_tracing
        
    else:
        raise NotImplementedError("module {} not implemented".format(name))
