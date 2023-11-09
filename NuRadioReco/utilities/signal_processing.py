import numpy as np
from scipy.signal.windows import hann


# TODO: put functions that are module specific into that module
def num_double_zeros(data, threshold=None, ave_shift=False):
    """if data is a numpy array, give number of points that have  zero preceded by a zero"""

    if ave_shift:
        data = data - np.average(data)

    if threshold is None:
        is_zero = data == 0
    else:
        is_zero = np.abs(data) < threshold

    bad = np.logical_and(is_zero[:-1], is_zero[1:])
    return np.sum(bad)


def half_hann_window(length, half_percent=None, hann_window_length=None):
    """produce a half-hann window. Note that this is different from a Hamming window."""
    if half_percent is not None:
        hann_window_length = int(length * half_percent)
    hann_window = hann(2 * hann_window_length)

    half_hann_widow = np.ones(length, dtype=np.double)
    half_hann_widow[:hann_window_length] = hann_window[:hann_window_length]
    half_hann_widow[-hann_window_length:] = hann_window[hann_window_length:]

    return half_hann_widow
