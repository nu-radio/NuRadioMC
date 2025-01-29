from NuRadioReco.framework.parameters import channelParameters as chp, stationParameters as stp
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import hilbert
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from NuRadioReco.modules.base.module import register_run


class stationLDAVariables:
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
            window size used for calculating the maximum peak to peak amplitude

        pad_length : int (default 500)
            padding length used for calculating the coherent sum

        channel_ids : array of int (default: [0,1,2,3])
            channels for which to calculate the variables

        """
        self.sum_chan = None
        self.__coincidence_window_size = coincidence_window_size
        self.__pad_length = pad_length
        self.__channel_ids = channel_ids


    @register_run()
    def run(self, event, station, detector, channel_id=0):

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

        channel_id: int
            reference channel for the coherent sum

        """
        station.set_parameter(stp.max_a, self.max_a(event, station, detector))
        self.avg_ch_snr(event, station, detector)
        station.set_parameter(stp.avg_ch_snr, self.avg_ch_snr(event, station, detector))
        self.coherent_sum(event, station, station.get_channel(channel_id))
        self.coherent_sum_step_by_step(event, station)
        station.set_parameter(stp.coherent_snr, self.coherent_snr(self.sum_chan))
        for ch_id in self.__channel_ids:
            ch = station.get_channel(ch_id)
            ch.set_parameter(chp.impulsive_value, self.impulsive_value(ch.get_trace()))
        return
    def end(self):
        pass

    def max_a(self, event, station, detector):
        maxaval = 0
        for channel in station.iter_channels():
            normalized_wf = channel.get_trace() / np.std(channel.get_trace())
            thismax = np.amax(
                self.maximum_peak_to_peak_amplitude(normalized_wf)
            )

            if thismax > maxaval:
                maxaval = thismax
        return maxaval

    def maximum_peak_to_peak_amplitude(self, trace):
        return maximum_filter1d(
            trace, self.__coincidence_window_size
        ) - minimum_filter1d(trace, self.__coincidence_window_size)

    def avg_ch_snr(self, event, station, detector):
        snrs = []
        for channel in station.iter_channels(use_channels=self.__channel_ids):
            split_array = np.array_split(channel.get_trace(), 4)
            rms_of_splits = np.std(split_array, axis=1)
            ordered_rmss = np.sort(rms_of_splits)
            lowest_two = ordered_rmss[:2]
            rms = np.mean(lowest_two)
            snr = np.amax(self.maximum_peak_to_peak_amplitude(channel.get_trace())) / (
                2 * rms
            )
            snrs.append(snr)
        avg_snr = np.mean(snrs)
        return avg_snr

    def coherent_snr(self, coherent_sum):

        split_array = np.array_split(coherent_sum, 4)
        rms_of_splits = np.std(split_array, axis=1)
        ordered_rmss = np.sort(rms_of_splits)

        lowest_two = ordered_rmss[:2]
        rms = np.mean(lowest_two)
        snr = np.amax(self.maximum_peak_to_peak_amplitude(coherent_sum)) / (
            2 * rms
        )

        return snr

    def coherent_sum_step_by_step(self, event, station):
        # Plot the four original waveforms before any alignment and save
        matplotlib.rcParams.update({"font.size": 20})
        plt.figure(figsize=(10, 6))
        for channel in station.iter_channels(use_channels=self.__channel_ids):
            plt.plot(channel.get_trace(), label=f'Original wf[{channel.get_id()}]')
        plt.title('Original Waveforms')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.savefig('original_waveforms.png')
        plt.show()
        
        # Iterate over each channel as a potential reference
        for ref_ch in station.iter_channels(use_channels = self.__channel_ids):
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # Create 2x2 grid for each step with a single reference
            fig.suptitle(f'Coherent Sum Steps with Reference: wf[{ref_ch.get_id()}]', fontsize = 16)
            
            sum_chan = np.pad(ref_ch.get_trace(), self.__pad_length, mode='constant')  # Pad reference waveform
            channels = [ch for ch in station.iter_channels(use_channels=self.__channel_ids) if ch.get_id() != ref_ch.get_id()]  # Exclude the reference channel
            
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

    def coherent_sum(self, event, station, ref_ch):

        self.sum_chan = ref_ch.get_trace()
        channels = [ch for ch in station.iter_channels(use_channels=self.__channel_ids) if ch.get_id() != ref_ch.get_id()]
                
        for idx, ch in enumerate(channels):
            cor = signal.correlate(np.abs(hilbert(self.sum_chan)), np.abs(hilbert(ch.get_trace())), mode = "full")
            lag = int(np.argmax((cor)) - (np.size(cor)/2.))
            
            aligned_wf = np.roll(ch.get_trace(), lag)
            self.sum_chan += aligned_wf
            
    def impulsive_value(self, volts):

        analytical_signal = hilbert(
            volts
        )  # compute analytic signal using hilbert transform from signal voltages
        envelope = np.abs(analytical_signal)
        maxv = np.argmax(envelope)
        self.maxspot = (
            maxv  ## index where the max voltage of the coherent sum is located
        )
        power_indexes = np.linspace(
            0, len(envelope) - 1, len(envelope)
        )  ## just a list of indices the same length as the array
        closeness = list(
            np.abs(power_indexes - maxv)
        )  ## create an array containing index distance to max voltage (lower the value, the closer it is)

        sorted_power = [x for _, x in sorted(zip(closeness, envelope))]
        cdf = np.cumsum(sorted_power)
        cdf = cdf / cdf[-1]

        cdf_avg = (np.mean(np.asarray([cdf])) * 2.0) - 1.0
        
        self.cdf_avg = cdf_avg

        if cdf_avg < 0:
            cdf_avg = 0.0

        return cdf_avg