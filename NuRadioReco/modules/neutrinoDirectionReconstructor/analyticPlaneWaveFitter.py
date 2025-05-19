import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
from scipy.spatial.transform import Rotation
from math import ceil
from radiotools import helper as hp
from NuRadioReco.utilities import units
import logging

logger = logging.getLogger('NuRadioReco.analyticPlaneWaveFitter')

SPEED_OF_LIGHT = scipy.constants.c * units.m / units.s # convert to NuRadio units

def find_threshold_crossing(channels, threshold, offset=5 * units.ns, min_amp=0, debug=False):
    """Find the time where the hilbert envelope of a trace first crosses some threshold

    Parameters
    ----------
    channels : list of `NuRadioReco.framework.channel.Channel` objects
        the channels to use in the reconstruction
    threshold : float | str
        If a float, the threshold value. Otherwise, the string "max" can be given,
        in which case the times of the hilbert envelope maxima are returned.
    offset : float or list of floats, optional
        Time at the start and end of each trace to exclude.
        If a list, the entries are the offsets for the start and end of the trace,
        respectively. By default, the first and last 5 ns of each trace are excluded,
        to avoid spurious peaks in the hilbert envelope resulting from the assumed
        periodicity of the trace.
    min_amp : float, optional
        Minimum amplitude for a trace to be considered valid.
        If the maximum amplitude of a channel does not exceed ``min_amp``,
        no threshold crossing will be returned for this channel.
    debug : bool, default False
        create some debug plots

    Returns
    -------
    threshold_times : np.ndarray of floats
        An array with the threshold crossing times for each channel.
        For channels that do not exceed ``threshold`` (or ``min_amp``),
        this will be ``np.nan``.
    """
    offset = list(offset)
    if len(offset) == 1:
        offset = (offset[0], offset[0])
    if debug:
        fig, axs = plt.subplots(3, 1, figsize=(4, 6))

    threshold_times = np.nan * np.zeros(len(channels))

    for i, channel in enumerate(channels):
        trace = channel.get_trace()
        sampling_rate = channel.get_sampling_rate()
        trace -= np.mean(trace)
        offset_samples = ceil(offset[0] * sampling_rate), ceil(offset[1] * sampling_rate)
        if offset_samples[1] == 0:
            offset_samples[1] = None # needed for correct numpy slicing

        hilbert_envelope = channel.get_hilbert_envelope_mag()
        if threshold == 'max':
            threshold_xing = np.argmax(hilbert_envelope[offset_samples[0]:-offset_samples[1]], keepdims=True)
        elif isinstance(threshold, str):
            raise ValueError(f'Argument `threshold` has value"{threshold}" but only a float or the string `max` are accepted.')
        else:
            threshold_xing = np.where(
                hilbert_envelope[offset_samples[0]:-offset_samples[1]] > threshold
            )[0]
        if np.max(hilbert_envelope) < min_amp:
            logger.debug("Reject - max amp {:.2f} < {:.2f}".format(np.max(hilbert_envelope), min_amp))
            continue
        if len(threshold_xing) == 0:
            logger.debug(
                "Reject - no threshold crossing (max amp {:.2f} < {:.2f})".format(
                    np.max(hilbert_envelope[offset_samples[0]:-offset_samples[1]]), threshold))
            continue

        threshold_t_sample = threshold_xing[0] + offset_samples[0]

        threshold_times[i] = channel.get_times()[threshold_t_sample]

        if debug:
            axs[i].plot(channel.get_times(), hilbert_envelope)
            axs[i].plot(channel.get_times(), trace)
            axs[i].axvline(threshold_times[i], ls=":", color='red')
            axs[i].set_xlim(threshold_times[i] - 50, threshold_times[i] + 150)
            axs[i].set_ylim(-.1*threshold, 5*threshold)
            axs[i].set_ylabel('amplitude')

    if debug:
        if len(threshold_times)==len(channels):
            axs[-1].set_xlabel('Time [ns]')
            plt.tight_layout()
            plt.show()
        else:
            plt.close()

    return np.asarray(threshold_times)


