from NuRadioReco.utilities import units, fft

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import numpy as np
import sys

import logging
logger = logging.getLogger("NuRadioReco.modules.channelSinewaveSubtraction")

"""
This module provides a class for continuous wave (CW) noise filtering using sine subtraction.
In contrast to the module channelCWNOtchFilter, which uses a notch filter to remove CW noise.
"""


class channelSinewaveSubtraction:
    """ Continuous wave (CW) filter module. Uses sine subtraction based on scipy curve_fit. """
    def __init__(self):
        pass

    def begin(self, peak_prominance=4, save_filtred_freqs=False):
        self.peak_prominance = peak_prominance
        self.save_filtred_freqs = [] if save_filtred_freqs else None

    def run(self, event, station, det=None):
        for channel in station.iter_channels():
            sampling_rate = channel.get_sampling_rate()

            trace = channel.get_trace()
            trace_fil = sinewave_subtraction(
                trace, sampling_rate=sampling_rate, peak_prominance=self.peak_prominance,
                saved_noise_freqs=self.save_filtred_freqs)

            channel.set_trace(trace_fil, sampling_rate)


def guess_amplitude(wf: np.ndarray, target_freq: float, sampling_rate: float = 3.2):
    """
    Estimate the amplitude of a specific harmonic in the waveform.

    Paramters
    ----------
    wf: np.ndarray
        Input waveform (1D array).
    target_freq:  float
        Target frequency for which to estimate amplitude.
    sampling_rate: float (default: 3.2)
        Sampling rate of the waveform (GHz).

    Returns
    --------
    ampl: float
        Estimated amplitude of the target frequency.
    """
    if wf.size == 0:
        raise ValueError("Input waveform is empty.")

    if target_freq < 0 or target_freq > sampling_rate / 2:
        raise ValueError("Target frequency is out of range (0 to Nyquist frequency).")

    fft_spectrum = fft.time2freq(wf, sampling_rate)
    frequencies = fft.freqs(len(wf), sampling_rate)

    # Find amplitude of the 50 Hz harmonic

    bin_index = np.argmin(np.abs(frequencies - target_freq))
    amplitude = np.abs(fft_spectrum[bin_index])

    return amplitude


def sinewave_subtraction(wf: np.ndarray, sampling_rate: float = 3.2, peak_prominance: float = 6.0, saved_noise_freqs: list = None):
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
        A list to store identified noise frequencies.

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
        return amplitude * np.cos(2 * np.pi * noise_frequency * t + phase)

    spec = abs(fft.time2freq(wf, sampling_rate))
    freqs = fft.freqs(len(wf), sampling_rate)

    # find noise frequencies:
    rms = np.sqrt(np.mean(np.abs(spec) ** 2))
    peak_idxs = np.where(np.abs(spec) > peak_prominance * rms)[0]

    noise_freqs = []

    # find mean CW freq bean
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
        corrected_waveform = wf.copy()
        for noise_freq in noise_freqs:

            ampl_guess = guess_amplitude(wf, noise_freq, sampling_rate)
            initial_guess = [ampl_guess, noise_freq, 0.01]

            # Fit the sinusoidal model to the waveform
            try:
                params, _ = curve_fit(sinusoid, t, wf, p0=initial_guess)
                estimated_amplitude, estimated_freq, estimated_phase = params

                # Generate the estimated CW noise
                estimated_cw_noise = estimated_amplitude * np.cos(
                    2 * np.pi * estimated_freq * t + estimated_phase
                )

                logger.info(f"Subtract sinewave with a frequency: {estimated_freq / units.MHz:.1f} MHz, "
                            f"an amplitude: {estimated_amplitude:.1e} V/GHz and a phase: {estimated_phase / units.deg:.1f} deg")

                # Subtract the estimated CW noise
                corrected_waveform -= estimated_cw_noise

                # Save the identified noise frequency
                if saved_noise_freqs is not None:
                    saved_noise_freqs.append(noise_freq)

            except RuntimeError:
                logger.error(f"Curve fitting failed for frequency: {noise_freq / units.MHz} MHz")

        return corrected_waveform
    else:
        return wf


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
    sub.begin()
    ev_num = 66

    logger.setLevel(logging.DEBUG)

    for event in rnog_reader.run():
        if event.get_id() == ev_num:
            station_id = event.get_station_ids()[0]
            station = event.get_station(station_id)
            #channel = station.get_channel(args.channel)

            fig, axs = plt.subplots(1, 2, figsize=(14, 6))
            plot_trace(station.get_channel(args.channel), axs[0], label="before", plot_kwargs={"lw": 2})
            plot_ft(station.get_channel(args.channel), axs[1], label="before", plot_kwargs={"lw": 2})

            sub.run(event, station, det=0)

            plot_trace(station.get_channel(args.channel), axs[0], label="after", plot_kwargs={"lw": 1})
            plot_ft(station.get_channel(args.channel), axs[1], label="after", plot_kwargs={"lw": 1})
            # save plot into the current dir
            current_dir = os.path.dirname(os.path.abspath(__file__))
            fig.savefig(current_dir + "/test_cw_filter", bbox_inches="tight")
