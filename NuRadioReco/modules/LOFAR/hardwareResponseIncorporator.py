from NuRadioReco.detector.LOFAR import analog_components
from NuRadioReco.modules.base.module import register_run
import numpy as np
import time
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger("LOFAR_hardwareResponseIncorporator")


class hardwareResponseIncorporator:
    """
    Incorporates the gains and losses induced by the LOFAR 
    hardware. This is partially taken from the ARA 
    hardwareResponseIncorporator.

    author: Karen Terveer

    """

    def __init__(self):
        self.__debug = False
        self.begin()

    def begin(self, debug=False):
        self.__debug = debug

    @register_run()
    def run(self, station, det, sim_to_data=False):
        """
        
        Incorporates the LOFAR signal chain (cable loss and RCU
        gain)

        Parameters
        ----------
        station: Station object
            The station whose channels noise shall be added to
        det: Detector object
            The detector description
        sim_to_data: bool
            set to True if working with simulated data to add the 
            signal chain to it. Set to False if working with
            measured data.

        """

        channels = station.iter_channels()

        for channel in channels:

            #get cable length of channel to use correct attenuation file
            cab_len = det.get_cable_type_and_length(station.get_id(),channel.get_id())[1]

            # fetch component responses
            frequencies = channel.get_frequencies()
            cable_response = analog_components.get_cable_response(frequencies,cable_length=int(cab_len))
            RCU_response = analog_components.get_RCU_response(frequencies)

            # calculate total system response. component responses are in dB, convert to linear scale
            system_response = np.power(10.0,(cable_response['attenuation']/10.0)) * np.power(10.0,(RCU_response['gain']/10.0))
            trace_fft = channel.get_frequency_spectrum()
                
            if sim_to_data:

                trace_after_system_fft = trace_fft * system_response
                # zero first bins to avoid DC offset
                trace_after_system_fft[0] = 0
                channel.set_frequency_spectrum(trace_after_system_fft, channel.get_sampling_rate())

            else:
                trace_before_system_fft = np.zeros_like(trace_fft)
                trace_before_system_fft[np.abs(system_response) > 0] = trace_fft[np.abs(system_response) > 0] / system_response[np.abs(system_response) > 0]
                channel.set_frequency_spectrum(trace_before_system_fft, channel.get_sampling_rate())


            if self.__debug == True:

                system_response_spectrum = np.abs(system_response)
                original_signal_spectrum = np.abs(trace_fft)

                if sim_to_data:

                    applied_signal_spectrum = np.abs(trace_after_system_fft)

                    # Plotting
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

                    # Plot system response
                    ax1.plot(frequencies, system_response_spectrum)
                    ax1.set_xlim(0.03,0.08)
                    ax1.set_yscale("log")
                    ax1.set_xlabel('frequency (GHz)')
                    ax1.set_ylabel('amplitude')
                    ax1.set_title('system response')

                    # Plot original and applied signal spectra

                    ax2.plot(frequencies, original_signal_spectrum, label='original simulated signal')
                    ax2.plot(frequencies, applied_signal_spectrum, label='system response applied')
                    ax2.set_yscale("log")
                    ax2.set_xlim(0.029,0.081)
                    ax2.set_xlabel('frequency (GHz)')
                    ax2.set_ylabel('amplitude')
                    ax2.set_title('signal')
                    ax2.legend()
                    plt.tight_layout()

                    plt.show()

                else:

                    applied_signal_spectrum = np.abs(trace_before_system_fft)

                    # Plotting
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

                    # Plot system response
                    ax1.plot(frequencies, system_response_spectrum)
                    ax1.set_xlabel('frequency (GHz)')
                    ax1.set_ylabel('amplitude')
                    ax1.set_title('system response')
                    ax1.set_xlim(0.03,0.08)
                    ax1.set_yscale("log")

                    # Plot original and applied signal spectra
                    ax2.plot(frequencies, original_signal_spectrum, label='original signal')
                    ax2.plot(frequencies, applied_signal_spectrum, label='system response applied')
                    ax2.set_xlabel('frequency (GHz)')
                    ax2.set_ylabel('amplitude')
                    ax2.set_xlim(0.029,0.081)
                    ax2.set_yscale("log")
                    ax2.set_title('signal')
                    ax2.legend()
                    plt.tight_layout()

                    plt.show()                 


