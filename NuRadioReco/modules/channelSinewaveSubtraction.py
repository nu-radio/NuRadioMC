from NuRadioReco.utilities import units, fft

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import numpy as np
import sys


"""
This module provides a class for continuous wave (CW) noise filtering using sine subtraction.
In contrast to the module channelCWNOtchFilter, which uses a notch filter to remove CW noise.
"""


def guess_amplitude(wf: np.ndarray, target_freq: float, sampling_rate: float = 3.2):
    """
        Estimate the amplitude of a specific harmonic in the waveform.

        Args:
            wf (np.ndarray): Input waveform (1D array).
            target_freq (float): Target frequency for which to estimate amplitude.
            sampling_rate (float): Sampling rate of the waveform (GHz).

        Returns:
            float: Estimated amplitude of the target frequency.
        """
    if wf.size == 0:
        raise ValueError("Input waveform is empty.")
    if target_freq < 0 or target_freq > sampling_rate / 2:
        raise ValueError("Target frequency is out of range (0 to Nyquist frequency).")

    fft_spectrum = np.fft.fft(wf)
    frequencies = np.fft.fftfreq(len(wf), d=1 / sampling_rate)

    # Find amplitude of the 50 Hz harmonic

    bin_index = np.argmin(np.abs(frequencies - target_freq))
    amplitude = np.abs(fft_spectrum[bin_index]) / len(wf)  # Normalize
    if bin_index > 0:
        amplitude *= 2  # Adjust for double-sided spectrum
    #print(f"Amplitude of {target_freq} Hz harmonic: {amplitude}")
    return amplitude


def sine_sub(wf: np.ndarray, sampling_rate: float = 3.2, peak_prominance: float = 6.0, saved_noise_freqs : list = None):
    """
    Perform sine subtraction on a waveform to remove CW noise.

    Args:
        wf (np.ndarray): Input waveform (1D array).
        sampling_rate (float): Sampling rate of the waveform (Hz).
        peak_prominence (float): Threshold for identifying prominent peaks in the FFT spectrum.
        saved_noise_freqs (list, optional): A list to store identified noise frequencies.

    Returns:
        np.ndarray: Corrected waveform with CW noise removed.
    """
    dt = 1 / sampling_rate # in ns
    t = np.arange(0, len(wf) * dt, dt) # in ns
    # zero meaning, just in case
    wf = wf - np.mean(wf)

    def sinusoid(t, amplitude, noise_frequency, phase):
        return amplitude * np.cos(2 * np.pi * noise_frequency * t + phase)
    # get fft spectrum
    fft = abs(ut.fft.time2freq(wf, 3.2))
    freqs = np.fft.rfftfreq(len(wf), dt)

    # find noise frequencies:
    rms = np.sqrt(np.mean(np.abs(fft) ** 2))
    peak_idxs = np.where(np.abs(fft) > peak_prominance * rms)[0]
    #noise_freqs = freqs[peak_idxs]
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

            A_guess = guess_amplitude(wf, noise_freq, sampling_rate)
            initial_guess = [A_guess, noise_freq, 0.01]

            # Fit the sinusoidal model to the waveform
            try:
                params, _ = curve_fit(sinusoid, t, wf, p0=initial_guess)
                estimated_amplitude, estimated_freq, estimated_phase = params

                # Generate the estimated CW noise
                estimated_cw_noise = estimated_amplitude * np.cos(
                    2 * np.pi * estimated_freq * t + estimated_phase
                )

                # Subtract the estimated CW noise
                corrected_waveform -= estimated_cw_noise

                # Save the identified noise frequency
                if saved_noise_freqs is not None:
                    saved_noise_freqs.append(noise_freq)
            except RuntimeError:
                print(f"Curve fitting failed for frequency: {noise_freq}")

    return corrected_waveform


def plot_ft(channel, ax, label=None, plot_kwargs=dict()):
    """
    Function to plot real frequency spectrum of given channel

    Parameters
    ----------
    channel : NuRadio channel class
        channel from which to get trace
    ax : matplotlib.axes
        ax on which to plot
    label : string
        plotlabel
    plot_kwargs : dict
        options for plotting
    """
    freqs = channel.get_frequencies()
    spec = channel.get_frequency_spectrum()

    legendloc = 2

    ax.plot(freqs, np.abs(spec), label=label, **plot_kwargs)
    ax.set_xlabel("freq / GHz")
    ax.set_ylabel("amplitude / V/GHz")
    ax.legend(loc=legendloc)


def plot_trace(channel, ax, fs=3.2e9 * units.Hz, label=None, plot_kwargs=dict()):
    """
    Function to plot trace of given channel

    Parameters
    ----------
    channel : NuRadio channel class
        channel from which to get trace
    ax : matplotlib.axes
        ax on which to plot
    fs : float, default = 3.2 Hz
        sampling frequency
    label : string
        plotlabel
    plot_kwargs : dict
        options for plotting
    """
    times = np.arange(2048) / fs / units.ns
    trace = channel.get_trace()

    legendloc = 2

    ax.plot(times, trace, label=label, **plot_kwargs)
    ax.set_xlabel("time / ns")
    ax.set_ylabel("trace / V")
    ax.legend(loc=legendloc)

class channelSineSubtraction:
    """ Continuous wave (CW) filter module. Uses sine subtraction based on scipy curve_fit  """
    def __init__(self):
        pass

    def begin(self, sampling_rate=3.2, peak_prominance=4, save_filtred_freqs=False):
        self.sampling_rate = sampling_rate
        self.peak_prominance = peak_prominance
        self.save_filtred_freqs = [] if save_filtred_freqs else None

    def run(self, event, station, det):
        for channel in station.iter_channels():
            fs = channel.get_sampling_rate()
            # freq = channel.get_frequencies()
            # spectrum = channel.get_frequency_spectrum()
            trace = channel.get_trace()
            trace_fil = sine_sub(trace, sampling_rate=self.sampling_rate,
                                 peak_prominance=self.peak_prominance, saved_noise_freqs=self.save_filtred_freqs)
            channel.set_trace(trace_fil, fs)


if __name__ == "__main__":
    import argparse
    import os
    from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
    import requests
    import aiohttp

    parser = argparse.ArgumentParser(prog="%(prog)s", usage="cw filter test")
    parser.add_argument("--station", type=int, default=13)
    parser.add_argument("--channel", type = int, default = 0)
    parser.add_argument("--run", type=int, default=104)

    args = parser.parse_args()


    data_dir = os.environ["RNO_G_DATA"] # used deep CR burn sample..
    rnog_reader = readRNOGData(log_level = logging.DEBUG)

    root_dirs = f"{data_dir}/station{args.station}/run{args.run}"
    rnog_reader.begin(root_dirs,
                      # linear voltage calibration
                      convert_to_voltage=True,
                      mattak_kwargs=dict(backend="uproot"))


    sub = channelSineSubtraction()
    sub.begin()
    ev_num=732

    for event in rnog_reader.run():
        if event.get_id() == ev_num:
            station_id = event.get_station_ids()[0]
            station = event.get_station(station_id)
            #channel = station.get_channel(args.channel)

            fig, axs = plt.subplots(1, 2, figsize=(14, 6))
            plot_trace(station.get_channel(args.channel), axs[0], label="before")
            plot_ft(station.get_channel(args.channel), axs[1], label="before")

            sub.run(event, station, det=0)

            plot_trace(station.get_channel(args.channel), axs[0], label="after")
            plot_ft(station.get_channel(args.channel), axs[1], label="after")
            # save plot into the current dir
            current_dir = os.path.dirname(os.path.abspath(__file__))
            fig.savefig(current_dir+"/test_cw_filter", bbox_inches="tight")
