from NuRadioReco.utilities import units, geometryUtilities as geo_utl, fft

from NuRadioReco.detector import detector
from NuRadioReco.framework.sim_station import SimStation

from scipy.signal.windows import hann
import numpy as np

import logging
logger = logging.getLogger('NuRadioReco.signal_processing')



def half_hann_window(length, half_percent=None, hann_window_length=None):
    """
    Produce a half-Hann window. This is the Hann window from SciPY with ones inserted in the middle to make the window
    `length` long. Note that this is different from a Hamming window.

    Parameters
    ----------
    length : int
        The desired total length of the window
    half_percent : float, default=None
        The percentage of `length` at the beginning **and** end that should correspond to half of the Hann window
    hann_window_length : int, default=None
        The length of the half the Hann window. If `half_percent` is set, this value will be overwritten by it.
    """
    if half_percent is not None:
        hann_window_length = int(length * half_percent)
    elif hann_window_length is None:
        raise ValueError("Either half_percent or half_window_length should be set!")
    hann_window = hann(2 * hann_window_length)

    half_hann_widow = np.ones(length, dtype=np.double)
    half_hann_widow[:hann_window_length] = hann_window[:hann_window_length]
    half_hann_widow[-hann_window_length:] = hann_window[hann_window_length:]

    return half_hann_widow


def add_cable_delay(station, det, sim_to_data=None, trigger=False, logger=None):
    """
    Add or subtract cable delay by modifying the ``trace_start_time``.

    Parameters
    ----------
    station: Station
        The station to add the cable delay to.

    det: Detector
        The detector description

    trigger: bool
        If True, take the time delay from the trigger channel response.
        Only possible if ``det`` is of type `rnog_detector.Detector`. (Default: False)

    logger: logging.Logger, default=None
        If set, use ``logger.debug(..)`` to log the cable delay.

    See Also
    --------
    NuRadioReco.modules.channelAddCableDelay.channelAddCableDelay : module that automatically applies / corrects for cable delays.
    """
    assert sim_to_data is not None, "``sim_to_data`` is None, please specify."

    add_or_subtract = 1 if sim_to_data else -1
    msg = "Add" if sim_to_data else "Subtract"

    if trigger and not isinstance(det, detector.rnog_detector.Detector):
        raise ValueError("Simulating extra trigger channels is only possible with the `rnog_detector.Detector` class.")

    for channel in station.iter_channels():

        if trigger:
            if not channel.has_extra_trigger_channel():
                continue

            channel = channel.get_trigger_channel()
            cable_delay = det.get_cable_delay(station.get_id(), channel.get_id(), trigger=True)

        else:
            # Only the RNOG detector has the argument `trigger`. Default is false
            cable_delay = det.get_cable_delay(station.get_id(), channel.get_id())

        if logger is not None:
            logger.debug(f"{msg} {cable_delay / units.ns:.2f}ns "
                        f"of cable delay to channel {channel.get_id()}")

        channel.add_trace_start_time(add_or_subtract * cable_delay)


def get_efield_antenna_factor(station, frequencies, channels, detector, zenith, azimuth, antenna_pattern_provider):
    """
    Returns the antenna response to a radio signal coming from a specific direction

    Parameters
    ----------

    station: Station
    frequencies: array of complex
        frequencies of the radio signal for which the antenna response is needed
    channels: array of int
        IDs of the channels
    detector: Detector
    zenith, azimuth: float, float
        incoming direction of the signal. Note that refraction and reflection at the ice/air boundary are taken into account
    antenna_pattern_provider: AntennaPatternProvider
    """

    efield_antenna_factor = np.zeros((len(channels), 2, len(frequencies)), dtype=complex)  # from antenna model in e_theta, e_phi
    for iCh, channel_id in enumerate(channels):
        zenith_antenna, t_theta, t_phi = geo_utl.fresnel_factors_and_signal_zenith(detector, station, channel_id, zenith)

        if zenith_antenna is None:
            logger.warning("Fresnel reflection at air-firn boundary leads to unphysical results, no reconstruction possible")
            return None

        logger.debug("angles: zenith {0:.0f}, zenith antenna {1:.0f}, azimuth {2:.0f}".format(
            np.rad2deg(zenith), np.rad2deg(zenith_antenna), np.rad2deg(azimuth)))
        antenna_model = detector.get_antenna_model(station.get_id(), channel_id, zenith_antenna)
        antenna_pattern = antenna_pattern_provider.load_antenna_pattern(antenna_model)
        ori = detector.get_antenna_orientation(station.get_id(), channel_id)
        VEL = antenna_pattern.get_antenna_response_vectorized(frequencies, zenith_antenna, azimuth, *ori)
        efield_antenna_factor[iCh] = np.array([VEL['theta'] * t_theta, VEL['phi'] * t_phi])

    return efield_antenna_factor


def get_channel_voltage_from_efield(
        station, electric_field, channels, detector,
        zenith, azimuth, antenna_pattern_provider, return_spectrum=True):
    """
    Returns the voltage traces that would result in the channels from the station's E-field.

    Parameters
    ----------

    station: Station
    electric_field: ElectricField
    channels: array of int
        IDs of the channels for which the expected voltages should be calculated
    detector: Detector
    zenith, azimuth: float
        incoming direction of the signal. Note that reflection and refraction
        at the air/ice boundary are already being taken into account.
    antenna_pattern_provider: AntennaPatternProvider
    return_spectrum: boolean
        if True, returns the spectrum, if False return the time trace
    """

    frequencies = electric_field.get_frequencies()
    spectrum = electric_field.get_frequency_spectrum()
    efield_antenna_factor = get_efield_antenna_factor(station, frequencies, channels, detector, zenith, azimuth, antenna_pattern_provider)
    if return_spectrum:
        voltage_spectrum = np.zeros((len(channels), len(frequencies)), dtype=complex)
        for i_ch, ch in enumerate(channels):
            voltage_spectrum[i_ch] = np.sum(efield_antenna_factor[i_ch] * np.array([spectrum[1], spectrum[2]]), axis=0)
        return voltage_spectrum
    else:
        voltage_trace = np.zeros((len(channels), 2 * (len(frequencies) - 1)), dtype=complex)
        for i_ch, ch in enumerate(channels):
            voltage_trace[i_ch] = fft.freq2time(
                np.sum(efield_antenna_factor[i_ch] * np.array([spectrum[1], spectrum[2]]), axis=0),
                electric_field.get_sampling_rate())

        return np.real(voltage_trace)
