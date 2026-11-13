"""
Function to replace the njit decorator. If numba is not available we just use a dummy decorator, otherwise @njit
"""
import logging
logger = logging.getLogger("NuRadioMC")

try:
    from numba import njit
    logger.debug("Numba loaded")
except ImportError as e:
    logger.debug("Numba not available", exc_info=e)
    def njit(*args, **kwargs):
        def dec(func):
            return func
        return dec