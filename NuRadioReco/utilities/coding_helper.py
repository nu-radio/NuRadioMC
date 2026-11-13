"""
This module provides helper functions for coding in the NuRadio framework.
"""

import numpy as np
from typing import Callable, Iterable


def array_from_function(function: Callable, iterable: Iterable, dtype=np.float64) -> np.ndarray:
    """
    applies a function to each element of an iterable and returns the result as a numpy array

    Parameters
    ----------
    function: Callable
        function that is applied to each element of the iterable
    iterable: Iterable
        elements the function is applied to
    dtype: numpy dtype, default np.float64
        data type of the resulting array if the function returns scalars;
        ignored if the function returns numpy arrays

    Returns
    -------
    np.ndarray
        stacked array with 1 dimension more than the output of the function
    """
    # the branch is decided by the type of the first element (assuming homogeneous elements).
    # for generators, the first element is replayed below so that nothing is consumed
    # without being used (the rest of the iterable stays lazily streamed into np.fromiter)
    iterator = iter(iterable)
    try:
        first = next(iterator)
    except StopIteration:
        return np.fromiter((), dtype=dtype)

    def mapped():
        yield function(first)
        yield from (map(function, iterator))

    # return a stacked array to ensure array dtype in output
    if isinstance(first, np.ndarray):
        return np.stack(np.fromiter(mapped(), dtype=object), dtype=dtype)
    return np.fromiter(mapped(), dtype=dtype)
