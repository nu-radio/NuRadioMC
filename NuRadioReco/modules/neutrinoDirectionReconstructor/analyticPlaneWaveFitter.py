import numpy as np
from NuRadioReco.utilities import units
import matplotlib.pyplot as plt
from scipy.constants import c as speed_of_light
speed_of_light *= units.m / units.s # convert to NuRadio units

import logging
logging.basicConfig()
logger = logging.getLogger('wind-reco')
logger.setLevel(logging.INFO)

def find_threshold_crossing(channels, threshold, offset=5, min_amp=.15, debug=False):
    """Find the time where the hilbert envelope of a trace first crosses some threshold
    
    Parameters
    ----------
    channels: list of NuRadioReco.framework.channel objects
        the channels to use in the reconstruction
    threshold: float
        the threshold value
    offset: int or list of ints
        number of samples at the start and end of each trace to exclude
        if a list, the entries are the offsets for the start and end of the trace,
        respectively
    min_amp: float
        minimum amplitude for all traces to be considered valid. Traces whose 
        maximum amplitude is less than min_amp are rejected
    debug: bool, default False
        create some debug plots
    
    Returns
    -------
    threshold_times: np.ndarray of floats
        An array with the threshold crossing times for each channel
    
    """
    offset = list(offset)
    if len(offset) == 1:
        offset += offset
    if debug:
        fig, axs = plt.subplots(3, 1, figsize=(4,6))
    threshold_times = []
    for i, channel in enumerate(channels):
        trace = channel.get_trace()
        sampling_rate = channel.get_sampling_rate()
        trace -= np.mean(trace)
        channel.set_trace(trace, sampling_rate)

        hilbert_envelope = channel.get_hilbert_envelope_mag()
        if threshold == 'pulse_max':
            threshold_xing = [np.argmax(hilbert_envelope) - offset[0]]
        else:
            threshold_xing = np.where(
                hilbert_envelope[offset[0]:-offset[1]] > threshold
            )[0]
        if np.max(hilbert_envelope) < min_amp:
            logger.debug("Reject - max amp {:.2f} < {:.2f}".format(np.max(hilbert_envelope),min_amp))
            break
        if len(threshold_xing) == 0:
            logger.debug(
                "Reject - no threshold xing (max amp {:.2f} < {:.2f})".format(
                    np.max(hilbert_envelope[offset[0]:-offset[1]]), threshold))
            break
        threshold_t = threshold_xing[0] + offset[0]
        if threshold_t < offset[0] + 5: # we extend the offset slightly to reject 'mid-pulse' fits
            logger.debug("Reject - threshold_t {:.2f} smaller than offset {:d}".format(threshold_t, offset[0]))
            break
            

        threshold_times.append(channel.get_times()[threshold_t])
        if debug:
            axs[i].plot(channel.get_times(), hilbert_envelope)
            axs[i].plot(channel.get_times(), trace)
            axs[i].axvline(threshold_times[i], ls=":", color='red')
            axs[i].set_xlim(threshold_times[i] - 50, threshold_times[i] + 150)
            axs[i].set_ylim(-.02, 5*threshold)
            axs[i].set_ylabel('amplitude')
    if debug:
        if len(threshold_times)==len(channels):
            axs[-1].set_xlabel('time (ns)')
            plt.tight_layout()
            plt.show()
        else:
            plt.close()
        
    return np.array(threshold_times)

def analytic_plane_wave_fitter(dt, pos, n_index=1.000293):
    """analytic plane wave fit
    
    Given three time delays ``dt`` and three positions 
    ``pos``, returns the analytic solution(s) to the
    plane wave fit.
    
    Parameters
    ----------
    dt: (3)-shaped np.array 
        the (relative) times of the signal arrival
    pos: (3, 3)-shaped np.array
        the 3D positions of the three observers
    n_index: float, default 1.
        the index of refraction
    
    Returns
    -------
    (theta, phi): tuple of floats
        zenith and azimuth of the analytic solution
    
    """
    if len(dt) > 3:
        logger.warning("System overdetermined, using only first three time delays & observers")
    #TODO - implement rotation so this is valid
    #even if the observers don't have the same z coord
    if not all(np.abs(pos[:,2] - pos[0,2]) <= 1e-8):
        logger.warning("Z coordinates of observers are not identical! Result will not be valid.")
    
    pos_xy = pos[1:3,:2] - pos[0:1, :2]
    ds = speed_of_light * np.array(dt) / n_index
    ds = ds[1:3] - ds[0]

    sol_vector = -np.linalg.inv(pos_xy) @ ds # - sign because we want the source direction
    sin_theta = np.linalg.norm(sol_vector)

    if sin_theta > 1:
        logger.warning("No valid solution!")
        return np.nan, np.nan

    return np.arcsin(sin_theta), np.arctan2(sol_vector[1], sol_vector[0])
