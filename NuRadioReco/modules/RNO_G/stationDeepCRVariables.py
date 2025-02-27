from NuRadioReco.framework.parameters import channelParameters as chp, stationParametersRNOG as stpRNOG
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import hilbert
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from NuRadioReco.modules.base.module import register_run
import NuRadioReco.modules.channelSignalReconstructor
from NuRadioReco.utilities import trace_utilities


class stationDeepCRVariables:
    """
    Module that calculates some variables for a Linear Discriminant Analysis of a specific event and stores them at the station level.
    """
    def __init__(self):
        pass

    def begin(self, coincidence_window_size = 6, pad_length = 500, channel_ids = [0,1,2,3]):
        """
        Parameters
        ----------

        coincidence_window_size : int (default: 6)
            Window size used for calculating the maximum peak to peak amplitude

        pad_length : int (default 500)
            Padding length used for calculating the coherent sum

        channel_ids : array of int (default: [0,1,2,3])
            Channels for which to calculate the variables

        """
        self.__coincidence_window_size = coincidence_window_size
        self.__pad_length = pad_length
        self.__channel_ids = channel_ids


    @register_run()
    def run(self, event, station, detector, ref_ch_id = 0, use_envelope = True):

        """
        Calculate LDA variables and add to the station object.

        Parameters
        ----------

        event: Event object
            The event for which the LDA variables should be calculated

        station: Station object
            The station for which the LDA variables should be calculated

        detector: Detector object
            The detector description

        ref_ch_id: int
            reference channel for the coherent sum

        """

        ref_ch = station.get_channel(ref_ch_id)
        ref_trace = ref_ch.get_trace()
        trace_set = [ch.get_trace() for ch in station.iter_channels(use_channels = self.__channel_ids) if ch.get_id() != ref_ch.get_id()]

        sum_trace = self.coherent_sum(trace_set, ref_trace, use_envelope)
        station.set_parameter(stpRNOG.coherent_snr, self.coherent_snr(sum_trace))
        return

    def end(self):
        pass

    def coherent_snr(self, coherent_sum):
        snr = np.amax(trace_utilities.maximum_peak_to_peak_amplitude(coherent_sum, self.__coincidence_window_size))
        snr /= trace_utilities.split_trace_noise_rms(coherent_sum, segments=4, lowest=2)
        snr /= 2
        return snr

    def coherent_sum_step_by_step(self, station):
        # Plot the four original waveforms before any alignment and save
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

    def coherent_sum(self, trace_set, ref_trace, use_envelope = False):
        sum_wf = ref_trace
        for idx, trace in enumerate(trace_set):
            if use_envelope:
                sig_ref = trace_utilities.get_hilbert_envelope(ref_trace)
                sig_i = trace_utilities.get_hilbert_envelope(trace)
            else:
                sig_ref = ref_trace
                sig_i = trace
            cor = signal.correlate(sig_ref, sig_i, mode = "full")
            lag = int(np.argmax((cor)) - (np.size(cor)/2.))

            aligned_wf = np.roll(trace, lag)
            sum_wf += aligned_wf
        return sum_wf
