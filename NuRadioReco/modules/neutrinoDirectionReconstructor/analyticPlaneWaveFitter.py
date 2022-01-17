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

def analytic_plane_wave_fitter(dt, pos, n_index=1.):
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
    (theta, phi): np.array of floats
        zenith and azimuth of the analytic solution
    
    """
    ds = speed_of_light * np.array(dt) / n_index
    ch0 = np.argmin(ds)
    ind = [i for i in range(3) if i != ch0]
    ds -= np.min(ds)
    pos_shifted = pos - pos[ch0][None]
    dch = np.linalg.norm(pos_shifted, axis=1)
    
    tau = np.array([ds[i] / dch[i] for i in ind])
    logger.debug("ds/dch: {:.3g}, {:.3g}".format(*tau))
    
    #TODO - implement rotation so this is valid
    #even if the observers don't have the same z coord
    if not all(np.abs(pos[:,2] - pos[0,2]) <= 1e-8):
        logger.warning("Z coordinates of observers are not identical! Result will not be valid.")
    
    dphi = [np.arctan2(pos_shifted[i, 1], pos_shifted[i, 0]) for i in ind]
#     print(dphi)
    dphi1 = dphi[0]
    dphi2 = dphi[0] - dphi[1]
    
    x1 = (np.cos(dphi2) - tau[1]/tau[0]) / np.sin(dphi2)
    x2 = (np.cos(dphi2) + tau[1]/tau[0]) / np.sin(dphi2)
    phi = np.arctan(np.array([x1, x2]))
    theta = np.arcsin(tau[0] / np.abs(np.cos(phi)))
    # we find two solutions, so need to check which one is real:
    mask1 = np.abs((np.sin(theta) * np.cos(phi) - tau[0])) < 1e-8
    mask2 = np.abs((np.sin(theta) * np.cos(phi + dphi2) - tau[1])) < 1e-8
    mask = mask1 & mask2
    
    logger.debug("phi   : {:.3g}, {:.3g}".format(*phi))
    logger.debug("theta : {:.3g}, {:.3g}".format(*theta))
    logger.debug("valid : {}, {}".format(*mask))
    # undo implicit rotation of ch1 to np.pi
    phi += dphi1 - np.pi
    phi = phi % (2*np.pi)
    if np.sum(mask) > 1:
        if abs(theta[0]-theta[1]) + abs(phi[1]-phi[0]) > 1e-8:
            raise ValueError("More than one solution found!")
    elif np.sum(mask) == 0:
        raise ValueError("No valid solutions")
    return np.array([theta[mask][0], phi[mask][0]])
