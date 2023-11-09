import numpy as np
from scipy.signal.windows import hann


# TODO: put functions that are module specific into that module
def half_hann_window(length, half_percent=None, hann_window_length=None):
    """produce a half-hann window. Note that this is different from a Hamming window."""
    if half_percent is not None:
        hann_window_length = int(length * half_percent)
    hann_window = hann(2 * hann_window_length)

    half_hann_widow = np.ones(length, dtype=np.double)
    half_hann_widow[:hann_window_length] = hann_window[:hann_window_length]
    half_hann_widow[-hann_window_length:] = hann_window[hann_window_length:]

    return half_hann_widow
