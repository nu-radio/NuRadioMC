"""
(old) implementation of ice models

Only returns one of two values depending on whether the depth is below
or above 0 (assumed to be the ice-air) interface.

.. deprecated:: 3.2.0
    Use the ice models of `NuRadioMC.utilities.medium` instead, e.g.
    `NuRadioMC.utilities.medium.get_station_ice_model` and
    `NuRadioMC.utilities.medium.get_refractive_index`.

"""
import logging

from NuRadioReco.utilities.logging import deprecated

logger = logging.getLogger('NuRadioReco.utilities.ice')


@deprecated(
    "NuRadioReco.utilities.ice.get_refractive_index is deprecated and will be removed in a future version. "
    "Use NuRadioMC.utilities.medium.get_station_ice_model and NuRadioMC.utilities.medium.get_refractive_index instead.",
    category=FutureWarning)
def get_refractive_index(depth, site='southpole'):
    """
    Get refractive index for depth

    For sites that are not at the poles, always returns the refractive index
    of air (1.000293). Otherwise, returns 1.3

    .. deprecated:: 3.2.0
        Use `NuRadioMC.utilities.medium.get_refractive_index` instead.

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
        if not site.lower() in ['southpole', 'mooresbay', 'summit', 'greenland', 'sp', 'simulation']:
            logger.warning(f"Site '{site}' unknown, assuming in-ice detector")
        if depth <= 0:
            return 1.3
        else:
            return 1.000293