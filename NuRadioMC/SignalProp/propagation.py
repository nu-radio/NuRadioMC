from __future__ import absolute_import, division, print_function

def get_propagation_module(name='analytic'):
    """
    wrapper around all propagation modules
    
    The function returns the python class of the 
    respective propagation module
    
    Parameters
    ----------
    name: string
        * analytic: analytic ray tracer
    """
    if(name=='analytic'):
        from NuRadioMC.SignalProp.analyticraytracing import ray_tracing
        return ray_tracing
    elif(name=='direct_ray'):
        from NuRadioMC.SignalProp.directRayTracing import directRayTracing
        return directRayTracing
        
    else:
        raise NotImplementedError("module {} not implemented".format(name))
