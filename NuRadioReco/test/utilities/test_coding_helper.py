"""
Test for the helper functions in ``NuRadioReco.utilities.coding_helper``.
"""
import numpy as np
import pytest

from NuRadioReco.utilities.coding_helper import (
    array_from_scalar_function,
    array_from_vectorized_function,
)


def scalar_f(x):
    """Return a plain scalar: square of ``x`` plus 1."""
    return x * x + 1.0


def vector_f(x):
    """Return a fixed-shape array of length 3 scaled by ``x``."""
    return np.array([x, x * 2, x * 3])


def vector_f_int(x):
    """Return an ``int64`` array of length 2 scaled by ``x``."""
    return np.array([x, x * 2], dtype=np.int64)


def ragged_f(x):
    """Return an array whose length depends on ``x`` (ragged)."""
    return np.ones(int(x))


class TestArrayFromScalarFunction:
    """Tests for :func:`array_from_scalar_function`."""

    def test_list_input(self):
        """Scalar function on a list input gives the expected 1-D array."""
        xs = [0.0, 1.0, 2.0, 3.0]
        result = array_from_scalar_function(scalar_f, xs)
        expected = np.array([1.0, 2.0, 5.0, 10.0])
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == np.float64
        assert result.ndim == 1

    def test_generator_input(self):
        """Scalar function on a generator input gives the same result."""
        gen = (float(i) for i in range(4))
        result = array_from_scalar_function(scalar_f, gen)
        expected = np.array([1.0, 2.0, 5.0, 10.0])
        np.testing.assert_array_equal(result, expected)

    def test_ndarray_input(self):
        """Scalar function on a numpy array input works."""
        xs = np.arange(5, dtype=np.float64)
        result = array_from_scalar_function(scalar_f, xs)
        np.testing.assert_array_equal(result, scalar_f(xs))

    def test_custom_dtype(self):
        """The dtype parameter is honoured."""
        xs = [1.0, 2.0, 3.0]
        result = array_from_scalar_function(scalar_f, xs, dtype=np.float32)
        assert result.dtype == np.float32

    def test_integer_dtype(self):
        """Integer dtype works for integer-valued scalar output."""
        xs = [1, 2, 3]
        result = array_from_scalar_function(lambda x: x * 2, xs, dtype=np.int32)
        assert result.dtype == np.int32
        np.testing.assert_array_equal(result, np.array([2, 4, 6], dtype=np.int32))

    def test_empty_input(self):
        """Empty input yields an empty array."""
        result = array_from_scalar_function(scalar_f, [])
        assert result.shape == (0,)
        assert result.dtype == np.float64

    def test_matches_list_comprehension(self):
        """Helper result matches a plain list comprehension for the same input."""
        xs = list(range(100))
        helper = array_from_scalar_function(scalar_f, xs)
        comprehension = np.array([scalar_f(x) for x in xs])
        np.testing.assert_array_equal(helper, comprehension)


class TestArrayFromVectorizedFunction:
    """Tests for :func:`array_from_vectorized_function`."""

    def test_list_input(self):
        """Vectorized function on a list input gives the expected 2-D array."""
        xs = [1.0, 2.0, 3.0]
        result = array_from_vectorized_function(vector_f, xs)
        expected = np.array([[1, 2, 3],
                             [2, 4, 6],
                             [3, 6, 9]], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)
        assert result.shape == (3, 3)
        assert result.dtype == np.float64

    def test_generator_input(self):
        """Vectorized function on a generator input gives the same result."""
        gen = (float(i) for i in range(1, 4))
        result = array_from_vectorized_function(vector_f, gen)
        expected = array_from_vectorized_function(vector_f, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result, expected)

    def test_ndarray_input(self):
        """Vectorized function on a numpy array input works."""
        xs = np.arange(1.0, 6.0)
        result = array_from_vectorized_function(vector_f, xs)
        assert result.shape == (5, 3)
        np.testing.assert_array_equal(result, np.stack([vector_f(x) for x in xs]))

    def test_custom_dtype(self):
        """The dtype parameter casts the stacked result."""
        xs = [1.0, 2.0, 3.0]
        result = array_from_vectorized_function(vector_f_int, xs, dtype=np.float32)
        assert result.dtype == np.float32

    def test_dtype_none_preserves_input(self):
        """Passing dtype=None preserves the dtype of the per-element arrays."""
        xs = [1.0, 2.0, 3.0]
        result = array_from_vectorized_function(vector_f_int, xs, dtype=None)
        assert result.dtype == np.int64

    def test_empty_input_raises(self):
        """Empty input raises a ValueError from numpy.stack."""
        with pytest.raises(ValueError, match="need at least one array to stack"):
            array_from_vectorized_function(vector_f, [])

    def test_ragged_raises(self):
        """Ragged per-element shapes raise a ValueError from numpy.stack."""
        xs = [1, 2, 3]
        with pytest.raises(ValueError, match="same shape"):
            array_from_vectorized_function(ragged_f, xs)

    def test_matches_list_comprehension(self):
        """Helper result matches np.stack of a list comprehension."""
        xs = list(range(100))
        helper = array_from_vectorized_function(vector_f, xs)
        comprehension = np.stack([vector_f(x) for x in xs])
        np.testing.assert_array_equal(helper, comprehension)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
