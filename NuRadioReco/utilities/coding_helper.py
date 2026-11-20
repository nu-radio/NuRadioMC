"""
This module provides helper functions for coding in the NuRadio framework.
"""

import numpy as np
from typing import Callable, Iterable


def array_from_scalar_function(function: Callable, iterable: Iterable, dtype=np.float64) -> np.ndarray:
    """
    Apply a scalar-output function to each element of an iterable and return
    the results as a 1-D numpy array.

    The function is applied once per element of ``iterable`` and must return
    a scalar. This is the efficient choice for scalar output: the values are
    written directly into a single typed buffer by ``numpy.fromiter`` and no
    intermediate list is built.

    Parameters
    ----------
    function: Callable
        function that is applied to each element of ``iterable``;
        must return a scalar
    iterable: Iterable
        elements the function is applied to (a list, ``numpy.ndarray``,
        generator, ...)
    dtype: numpy dtype, default np.float64
        data type of the resulting array

    Returns
    -------
    np.ndarray
        1-D array of shape ``(len(iterable),)`` with the given ``dtype``

    Notes
    -----
    For functions that return same-shaped arrays, use
    :func:`array_from_vectorized_function` instead, which stacks the
    per-element results on a new leading axis.
    """
    return np.fromiter(map(function, iterable), dtype=dtype)


def array_from_vectorized_function(function: Callable, iterable: Iterable, dtype=np.float64) -> np.ndarray:
    """
    Apply a function returning same-shaped arrays to each element of an
    iterable and return the results as a numpy array stacked on a new leading
    axis.

    The function is applied once per element of ``iterable`` and must return
    an array (or a list / tuple that can be converted to one) of the **same
    shape** for every element. The per-element results are stacked along a new
    first axis, so the output has 1 dimension more than a single call to
    ``function``.

    Parameters
    ----------
    function: Callable
        function that is applied to each element of ``iterable``;
        must return an array (or array-like) of the same shape for each element
    iterable: Iterable
        elements the function is applied to (a list, ``numpy.ndarray``,
        generator, ...)
    dtype: numpy dtype, default np.float64
        data type of the stacked result; the per-element arrays are cast to
        this type when stacking. Pass ``None`` to keep the dtype of the
        arrays returned by ``function``.

    Returns
    -------
    np.ndarray
        stacked array with 1 dimension more than the output of a single call
        to ``function``

    Notes
    -----
    For scalar-output functions use :func:`array_from_scalar_function`
    instead: it is faster because it avoids building an intermediate object
    array and an explicit ``numpy.stack``.
    """
    return np.stack(np.fromiter(map(function, iterable), dtype=object), dtype=dtype)
