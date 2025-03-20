"""
(old) implementation of ice models

Only returns one of two values depending on whether the depth is below
or above 0 (assumed to be the ice-air) interface.

.. Warning::
    This function is used internally in some modules, but should not
    be used in new user code. Please use the `NuRadioMC.utilities.medium`
    module instead

"""
import logging

logger = logging.getLogger('NuRadioReco.utilities.ice')

def get_refractive_index(depth, site='southpole'):
    """
    Get refractive index for depth

    For sites that are not at the poles, always returns the refractive index
    of air (1.000293). Otherwise, returns 1.3

    .. Warning::
        This function is only used internally. New user code should
        use the ice models in `NuRadioMC.utilities.medium` instead

    Parameters
    ----------
    depth : float
        The depth
    site : str, optional
        The site to use. For sites on land (not in-ice),
        the refractive index returned is always that for air.

    Returns
    -------
    n : float
        The refractive index. For land-based sites,
        this is always n_air=1.000293; for in-ice sites,
        returns n_ice=1.3 or n_air depending on the depth.

    """
    if site.lower() in ['lofar', 'auger', 'ska']:
        return 1.000293
    else:
        if not site.lower() in ['southpole', 'mooresbay', 'summit', 'greenland', 'sp']:
            logger.warning(f"Site '{site}' unknown, assuming in-ice detector")
        if depth <= 0:
            return 1.3
        else:
            return 1.000293