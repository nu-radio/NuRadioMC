from NuRadioReco.framework.parameters import stationParametersRNOG as stpRNOG

from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import trace_utilities, units

import matplotlib
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger('NuRadioReco.RNO_G.stationCoherentlySummedWaveforms')


class stationCoherentlySummedWaveforms:
    """
    Generates a coherently-summed waveform (CSW) and calculates its signal-to-noise-ratio (SNR).

    When multiple waveforms and a referance waveform are given,
    one can find the cross correlation between each waveform and the referance waveform,
    then each waveform will be rolled to line up the signal position with the referance
    based on the time lag when both waveforms are most correlated,
    and then all waveforms including the reference will be summed up
    to become a coherently-summed waveform (CSW).
    Thermal noise is random so it scales with the square root of added waveforms while coherent signals
    will added up linearly. Once summed the CSW is treated as a regular waveform and different analysis variables
    are calculated.
    """

    def __init__(self):
        """(Unused)"""
        pass

    def begin(self, coincidence_window_size=6 * units.ns, pad_length=500, channel_ids=[0, 1, 2, 3]):
        """
        Parameters
        ----------
        coincidence_window_size: float (default: 6 * units.ns)
            Window size used for calculating the maximum peak to peak amplitude in nanoseconds

        pad_length: int (default: 500)
            Padding length used for calculating the coherent sum

        channel_ids: array of int (default: [0, 1, 2, 3])
            Channels for which to calculate the variables
        """
        self.__coincidence_window_size = coincidence_window_size
        self.__pad_length = pad_length
        self.__channel_ids = channel_ids


    @register_run()
    def run(self, evt, station, det, ref_ch_id=0, use_envelope=True):
        """
        Calculate the SNR of the coherently-summed waveform and add to the station object.

        Parameters
        ----------
        evt, station, det
            Event, Station, Detector
        ref_ch_id: int (default: 0)
            Reference channel for the coherent sum
        use_envelope: bool (default: True)
            If use Hilbert envelopes to find the cross correlation or not
        """

        ref_ch = station.get_channel(ref_ch_id)
        ref_trace = ref_ch.get_trace()
        trace_set = [ch.get_trace() for ch in station.iter_channels(use_channels = self.__channel_ids) if ch.get_id() != ref_ch.get_id()]

        coincidence_window_size_bins_ref = int(round(self.__coincidence_window_size * ref_ch.get_sampling_rate()))
        if coincidence_window_size_bins_ref < 2:
            logger.warning(f"Coincidence window size of {coincidence_window_size_bins_ref} samples is too small for channel {ref_ch.get_id()}.")

        sum_trace = trace_utilities.get_coherent_sum(trace_set, ref_trace, use_envelope)
        rms = trace_utilities.get_split_trace_noise_RMS(sum_trace, segments=4, lowest=2)
        snr = trace_utilities.get_signal_to_noise_ratio(sum_trace, rms, window_size=coincidence_window_size_bins_ref)
        impulsivity = trace_utilities.get_impulsivity(sum_trace)
        entropy = trace_utilities.get_entropy(sum_trace)
        kurtosis = trace_utilities.get_kurtosis(sum_trace)
        
        station.set_parameter(stpRNOG.coherent_snr, snr)
        station.set_parameter(stpRNOG.coherent_impulsivity, impulsivity)
        station.set_parameter(stpRNOG.coherent_entropy, entropy)
        station.set_parameter(stpRNOG.coherent_kurtosis, kurtosis)

    def end(self):
        """(Unused)"""
        pass

    def coherent_sum_step_by_step(self, station):
        """ Plot the four original waveforms before any alignment and save """

        matplotlib.rcParams.update({"font.size": 20})
        plt.figure(figsize = (10, 6))
        for channel in station.iter_channels(use_channels = self.__channel_ids):
            plt.plot(channel.get_trace(), label = f'Original wf[{channel.get_id()}]')
        plt.title('Original Waveforms')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.savefig('original_waveforms.png')
        plt.show()

        # Iterate over each channel as a potential reference
        for ref_ch in station.iter_channels(use_channels = self.__channel_ids):
            fig, axs = plt.subplots(2, 2, figsize = (12, 10))  # Create 2x2 grid for each step with a single reference
            fig.suptitle(f'Coherent Sum Steps with Reference: wf[{ref_ch.get_id()}]', fontsize = 16)

            sum_chan = np.pad(ref_ch.get_trace(), self.__pad_length, mode = 'constant')  # Pad reference waveform
            channels = [ch for ch in station.iter_channels(use_channels = self.__channel_ids) if ch.get_id() != ref_ch.get_id()]  # Exclude the reference channel

            # Initialize the coherent sum with the padded reference waveform
            current_sum = sum_chan.copy()

            for step, ch in enumerate(channels):
                ax = axs[step // 2, step % 2]  # Select subplot for each step

                # Perform full-range cross-correlation to find the necessary shift
                cor = signal.correlate(current_sum[self.__pad_length:-self.__pad_length], ch.get_trace(), mode = 'full')
                shift = np.argmax(cor) - (len(ch.get_trace()) - 1)  # Calculate shift based on maximum correlation

                # Pad and shift the waveform to align with the reference sum
                padded_wf = np.pad(ch.get_trace(), self.__pad_length, mode = 'constant')
                aligned_wf = np.roll(padded_wf, shift)

                # Plot the current coherent sum before adding the next waveform
                ax.plot(current_sum[self.__pad_length:-self.__pad_length], label = 'Current Coherent Sum', linestyle = '--')
                ax.plot(aligned_wf[self.__pad_length:-self.__pad_length], label = f'Next wf[{ch.get_id()}]', alpha = 0.7)

                # Add the aligned waveform to the current sum
                current_sum += aligned_wf

                # Customize plot
                ax.legend(loc = 'upper right')
                ax.set_xlabel('Sample')
                ax.set_ylabel('Amplitude')
                ax.set_title(f'Step {step + 1}: Adding wf[{ch.get_id()}]')

            # Save each coherent summing figure for the current reference
            plt.tight_layout(rect = [0, 0, 1, 0.96])
            plt.savefig(f'coherent_sum_steps_reference_{ref_ch.get_id()}.png')
            plt.show()
