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
    threshold : float
        the threshold value
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
        offset += offset
    if debug:
        fig, axs = plt.subplots(3, 1, figsize=(4,6))

    threshold_times = np.nan * np.zeros(len(channels))

    for i, channel in enumerate(channels):
        trace = channel.get_trace()
        sampling_rate = channel.get_sampling_rate()
        trace -= np.mean(trace)
        offset_samples = ceil(offset[0] * sampling_rate), ceil(offset[1] * sampling_rate)
        if offset_samples[1] == 0:
            offset_samples[1] = None # needed for correct numpy slicing

        hilbert_envelope = channel.get_hilbert_envelope_mag()
        if threshold == 'pulse_max':
            threshold_xing = [np.argmax(hilbert_envelope) - offset_samples[0]]
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
    dt : (3)-shaped np.array
        the (relative) times of the signal arrival
    pos : (3, 3)-shaped np.array
        the 3D positions of the three observers
    n_index : float, default 1.
        the index of refraction
    
    Returns
    -------
    (theta, phi) : tuple of floats
        zenith and azimuth of the analytic solution
    
    Notes
    -----
    Note that the solution returned is not unique; mirroring the direction
    in the plane formed by the three observer positions also gives
    a valid solution. If this plane is the x-y plane (all observers have the same
    z coordinate), the solution coming from above (zenith < pi/2) is returned.

    """
    if len(dt) > 3:
        logger.warning("System overdetermined, using only first three time delays & observers")

    dpos = pos - pos[0:1]
    rot = None

    # If the observers don't all have the same z coord,
    # we perform a rotation such that they do
    if not all(np.abs(dpos[:,2]) <= 1e-8):
        rot_angle, phi_dpos = hp.cartesian_to_spherical(*np.cross(dpos[1], dpos[2]))
        rot = Rotation.from_rotvec(
            np.sign(rot_angle - np.pi/2) * rot_angle
            * hp.spherical_to_cartesian(np.pi/2, phi_dpos + np.pi/2))
        pos_xy = rot.apply(dpos)[1:3, 0:2]
    else:
        pos_xy = dpos[1:3, 0:2]
    
    ds = SPEED_OF_LIGHT * np.array(dt) / n_index
    ds = ds[1:3] - ds[0]

    sol_vector = -np.linalg.inv(pos_xy) @ ds # - sign because we want the source direction
    sin_theta = np.linalg.norm(sol_vector)

    if sin_theta > 1:
        logger.warning("No valid solution!")
        return np.nan, np.nan

    if rot is None:
        return np.arcsin(sin_theta), np.arctan2(sol_vector[1], sol_vector[0])
    else: # we have rotated our coordinate system, so we should rotate back
        theta_rot, phi_rot = np.arcsin(sin_theta), np.arctan2(sol_vector[1], sol_vector[0])
        v_rot = hp.spherical_to_cartesian(theta_rot, phi_rot)
        zenith, azimuth = hp.cartesian_to_spherical(*rot.apply(v_rot, inverse=True))
        return zenith, azimuth

