from scipy.signal import hilbert
import numpy as np

def envelope(channel_signals):
    for channel in channel_signals.keys():
        channel_signals[channel] = np.abs(hilbert(channel_signals[channel]))

    return channel_signals
