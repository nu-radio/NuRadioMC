from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units, fft

# For typing
import NuRadioReco.framework.event
import NuRadioReco.framework.station

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import lfilter
import numpy as np
import sys
import time
import logging
logger = logging.getLogger("NuRadioReco.modules.channelSinewaveSubtraction")

"""
This module provides a class for continuous wave (CW) noise filtering using sine subtraction.
In contrast to the module channelCWNOtchFilter, which uses a notch filter to remove CW noise.
"""


class channelSinewaveSubtraction:
    """ Continuous wave (CW) filter module. Uses sine subtraction based on scipy curve_fit. """
    def __init__(self):
        self.freq_band = None
        self.save_filtered_freqs = None
        self.begin()

    def begin(self, save_filtered_freqs: bool = False, freq_band: tuple[float, float] = (0.1, 0.7)) -> None:
        """
        Initialize the CW filter module.

        Parameters
        ----------
        save_filtered_freqs: bool (default: False)
            Flag to save the identified noise frequencies for each channel.
        freq_band: tuple (default: (0.1, 0.7))
            Frequency band to calculate baseline RMS of fft spectrum. Used to identify noise peaks.
            0.1 to 0.7 GHz is the default for RNO-G, based on bandpass.

        """
        self.save_filtered_freqs = [] if save_filtered_freqs else None
        self.freq_band = freq_band

    @register_run()
    def run(self, event: NuRadioReco.framework.event.Event, station: NuRadioReco.framework.station.Station,
            det=None, peak_prominence: float = 4.0) -> None:
        """
        Run the CW filter module on a given event and station. Removes all the CW peaks > peak_prominence * RMS.

        Parameters
        ----------
        event: `NuRadioReco.framework.event.Event`
            Event object to process.
        station: `NuRadioReco.framework.station.Station`
            Station object to process.
        det: `NuRadioReco.detector.detector.Detector` (default: None)
            Detector object to process.
        peak_prominence: float (default: 4.0)
            Threshold for identifying prominent peaks in the FFT spectrum.
        """
        for channel in station.iter_channels():
            sampling_rate = channel.get_sampling_rate()
            trace = channel.get_trace()
            trace_fil = sinewave_subtraction(
                trace, peak_prominence, sampling_rate=sampling_rate,
                saved_noise_freqs=self.save_filtered_freqs, freq_band=self.freq_band)

            channel.set_trace(trace_fil, sampling_rate)

    def get_filtered_frequencies(self):
        """ Get the list of identified noise frequencies for each channel. """
        return self.save_filtered_freqs


def guess_amplitude(wf: np.ndarray, target_freq: float, sampling_rate: float = 3.2):
    """
    Estimate the amplitude of a specific harmonic in the waveform.

    Parameters
    ----------
    wf: np.ndarray
        Input waveform (1D array).
    target_freq:  float
        Target frequency (GHz) for which to estimate amplitude.
    sampling_rate: float (default: 3.2)
        Sampling rate of the waveform (GHz).

    Returns
    -------
    ampl: float
        Estimated amplitude of the target frequency.
    """
    if wf.size == 0:
        raise ValueError("Input waveform is empty.")

    if target_freq < 0 or target_freq > sampling_rate / 2:
        raise ValueError("Target frequency is out of range (0 to Nyquist frequency).")

    frequencies = fft.freqs(len(wf), sampling_rate)

    # Here we intentionally use a different FFT normalization which retains the amplitude
    # in the time domain.
    amplitude_spectrum = np.abs(np.fft.rfft(wf, sampling_rate) * 2 / len(wf))

    bin_index = np.argmin(np.abs(frequencies - target_freq))
    amplitude = amplitude_spectrum[bin_index]

    return amplitude


def guess_amplitude_iir(wf: np.ndarray, target_freq: float, sampling_rate: float = 3.2):
    """
    Estimate the amplitude of a specific frequency using an IIR filter representation of Goertzel.

    Parameters
    ----------
    wf: np.ndarray
        Input waveform (1D array).
    target_freq: float
        Target frequency (GHz) to analyze.
    sampling_rate: float (default: 3.2)
        Sampling rate of the waveform (GHz).

    Returns
    -------
    amplitude: float
        Estimated amplitude at the target frequency.
    """
    if np.any(np.isnan(wf)):
        raise ValueError("Input signal contains NaNs!")

    N = len(wf)  # Number of samples
    k = int(0.5 + (N * target_freq / sampling_rate))  # Frequency bin index
    omega = (2.0 * np.pi * k) / N  # Angular frequency
    scaling_factor = N / 2.0

    # IIR filter coefficients derived from Goertzel's difference equation
    b = [1.0, 0, 0.0]  # Numerator coefficients
    a = [1.0, -2.0 * np.cos(omega), 1.0]  # Denominator coefficients
    # Apply the filter
    filtered_signal = lfilter(b, a, wf)
    # Extract last two values for amplitude estimation
    s_prev = filtered_signal[-1]
    s_prev2 = filtered_signal[-2]

    # Compute real and imaginary parts of the signal at the target frequency
    real = s_prev - s_prev2 * np.cos(omega)
    imag = s_prev2 * np.sin(omega)

    # Compute magnitude (amplitude)
    amplitude = np.sqrt(real**2 + imag**2) / scaling_factor
    return amplitude


def guess_phase(fft_spec: np.ndarray, freqs: np.ndarray, target_freq: float):
    """
    Estimate the phase of a specific frequency in the FFT spectrum.

    Parameters
    ----------
    fft_spec: np.ndarray
        FFT spectrum of the waveform.
    freq: np.ndarray
        Frequency array corresponding to the FFT spectrum.
    target_freq: float
        Target frequency (GHz) for which to estimate phase.
    sampling_rate: float (default: 3.2)
        Sampling rate of the waveform (GHz).

    Returns
    -------
    phase: float
        Estimated phase of the target frequency.
    """
    # Find phase of the target frequency
    bin_index = np.argmin(np.abs(freqs - target_freq))
    phase = np.angle(fft_spec[bin_index])

    return phase


def sinewave_subtraction(wf: np.ndarray, peak_prominence: float = 4.0, sampling_rate: float = 3.2,
                         saved_noise_freqs: list = None, freq_band: tuple = (0.1, 0.7)):
    """
    Perform sine subtraction on a waveform to remove CW noise.

    Parameters
    ----------
    wf: np.ndarray
        Input waveform (1D array).
    sampling_rate: float (default: 3.2)
        Sampling rate of the waveform (GHz).
    peak_prominance: float (default: 6.0)
        Threshold for identifying prominent peaks in the FFT spectrum.
    saved_noise_freqs: list (default: None)
        A list to store identified noise frequencies for each channel.
    freq_band: tuple (default for RNO-g: (0.1, 0.7))
        Frequency band to calculate baseline RMS of fft spectrum. Used to identify noise peaks.

    Returns
    -------
    np.ndarray
        Corrected waveform with CW noise removed.
    """


    dt = 1 / sampling_rate # in ns
    t = np.arange(0, len(wf) * dt, dt) # in ns

    # zero meaning, just in case
    wf = wf - np.mean(wf)

    def sinusoid(t, amplitude, noise_frequency, phase):
        return amplitude * np.sin(2 * np.pi * noise_frequency * t + phase + np.pi/2)

    spec_complex = fft.time2freq(wf, sampling_rate) # need later to estimate phase

    spec = abs(spec_complex)
    freqs = fft.freqs(len(wf), sampling_rate)
    # find total power of the original waveform
    power_orig = np.sum(spec ** 2)

    # find noise frequencies:

    # frequency range for RMS calculation, defined by bandpass
    f_min, f_max = freq_band

    # Mask frequencies within the range
    band_mask = (freqs >= f_min) & (freqs <= f_max)
    filtered_spec = spec[band_mask]

    # Compute RMS in the selected frequency band
    rms_band = np.sqrt(np.mean(filtered_spec ** 2))

    # Find noise peaks based on this band-limited RMS
    peak_idxs = np.where(spec > peak_prominence * rms_band)[0]


    noise_freqs = []
    corrected_waveform = wf.copy()
    # find mean CW freq bin
    if len(peak_idxs) > 0:

        # Initialize a list to hold groups of neighboring peak indices
        group = [peak_idxs[0]]

        # Loop through the remaining peak indices to group neighboring peaks
        for i in range(1, len(peak_idxs)):
            if peak_idxs[i] - peak_idxs[i - 1] == 1:  # If the peak is neighboring the previous one
                group.append(peak_idxs[i])
            else:
                # Calculate the mean frequency for the current group of neighbors
                noise_freqs.append(np.mean(freqs[group]))
                # Start a new group with the current peak
                group = [peak_idxs[i]]

        # Don't forget to append the last group
        if group:
            noise_freqs.append(np.mean(freqs[group]))

        # Convert the list to a NumPy array (optional, if you prefer an array)
        noise_freqs = np.array(noise_freqs)

        if saved_noise_freqs is not None:

            saved_noise_freqs.append(noise_freqs)

        for noise_freq in noise_freqs:

            ampl_guess = guess_amplitude_iir(wf, noise_freq, sampling_rate)
            phase = guess_phase(spec_complex, freqs, noise_freq)

            initial_guess = [ampl_guess, noise_freq, phase]
            # Fit the sinusoidal model to the waveform
            try:

                params, covariance = curve_fit(sinusoid, t, wf, p0=initial_guess)
                # Check if any parameters are NaN or Inf
                if np.any(np.isnan(params)) or np.any(np.isinf(params)):
                    raise RuntimeError("Fit returned invalid parameters.")

                estimated_amplitude, estimated_freq, estimated_phase = params

                # Check if the covariance matrix is invalid
                if np.all(np.isinf(covariance)) or np.all(np.isnan(covariance)):
                    raise RuntimeError("Fit covariance matrix is invalid, fit may not have converged.")

                # Generate the estimated CW noise
                estimated_cw_noise = sinusoid(t, estimated_amplitude, estimated_freq,estimated_phase)

                logger.info(f"Subtract sinewave with a frequency: {estimated_freq / units.MHz:.1f} MHz, "
                            f"an amplitude: {estimated_amplitude:.1e} V/GHz and a phase: {estimated_phase / units.deg:.1f} deg")

                # Subtract the estimated CW noise
                corrected_waveform -= estimated_cw_noise
                power_after_subtraction = np.sum(abs(fft.time2freq(corrected_waveform, sampling_rate)) ** 2)
                logger.info(f"Power reduction: {100 * (1 - power_after_subtraction / power_orig):.1f}%")

                if power_orig < power_after_subtraction:
                    logger.warning("Power increased after subtraction. Skipping this frequency.")
                    corrected_waveform += estimated_cw_noise
                    raise RuntimeError("Power increased after subtraction. Reverse subtraction.")

            except RuntimeError:
                logger.error(f"Curve fitting failed for frequency: {noise_freq / units.MHz} MHz")

    else:
        saved_noise_freqs.append([])

    return corrected_waveform


def plot_ft(channel, ax, label=None, plot_kwargs=dict()):
    """
    Function to plot real frequency spectrum of given channel

    Parameters
    ----------
    channel: `NuRadioReco.framework.channel.Channel`
        Channel from which to get trace
    ax: matplotlib.axes
        ax on which to plot
    label: string
        plotlabel
    plot_kwargs: dict
        options for plotting
    """
    freqs = channel.get_frequencies()
    spec = channel.get_frequency_spectrum()

    legendloc = 2

    ax.plot(freqs / units.MHz, np.abs(spec), label=label, **plot_kwargs)
    ax.set_xlabel("freq / MHz")
    ax.set_ylabel("amplitude / V/GHz")
    ax.set_yscale("log")
    ax.set_ylim(np.mean(np.abs(spec)) / 100, None)
    ax.legend(loc=legendloc)


def plot_trace(channel, ax, label=None, plot_kwargs=dict()):
    """
    Function to plot trace of given channel.

    Parameters
    ----------
    channel: `NuRadioReco.framework.channel.Channel`
        Channel from which to get trace
    ax: matplotlib.axes
        ax on which to plot
    fs: float, default = 3.2 Hz
        sampling frequency
    label: string
        plotlabel
    plot_kwargs: dict
        options for plotting
    """
    times = channel.get_times()
    trace = channel.get_trace()

    ax.plot(times, trace, label=label, **plot_kwargs)
    ax.set_xlabel("time / ns")
    ax.set_ylabel("trace / V")
    ax.legend(loc=2)


if __name__ == "__main__":

    from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
    import argparse
    import os

    parser = argparse.ArgumentParser(prog="%(prog)s", usage="cw filter test")
    parser.add_argument("--station", type=int, default=13)
    parser.add_argument("--channel", type = int, default = 0)
    parser.add_argument("--run", type=int, default=104)

    args = parser.parse_args()

    data_dir = os.environ["RNO_G_DATA"] # used deep CR burn sample..
    rnog_reader = readRNOGData(log_level = logging.INFO)

    root_dirs = f"{data_dir}/station{args.station}/run{args.run}"
    rnog_reader.begin(root_dirs,
                      # linear voltage calibration
                      convert_to_voltage=False,
                      mattak_kwargs=dict(backend="uproot"))

    sub = channelSinewaveSubtraction()
    sub.begin(save_filtered_freqs=True)
    ev_num = 66

    logger.setLevel(logging.DEBUG)

    for event in rnog_reader.run():
        if event.get_id() == ev_num:
            station_id = event.get_station_ids()[0]
            station = event.get_station(station_id)

            fig, axs = plt.subplots(1, 2, figsize=(14, 6))
            plot_trace(station.get_channel(args.channel), axs[0], label="before", plot_kwargs={"lw": 2})
            plot_ft(station.get_channel(args.channel), axs[1], label="before", plot_kwargs={"lw": 2})

            sub.run(event, station, det=0)

            plot_trace(station.get_channel(args.channel), axs[0], label="after", plot_kwargs={"lw": 1})
            plot_ft(station.get_channel(args.channel), axs[1], label="after", plot_kwargs={"lw": 1})
            # save plot into the current dir
            current_dir = os.path.dirname(os.path.abspath(__file__))
            fig.savefig(current_dir + "/test_cw_filter", bbox_inches="tight")
