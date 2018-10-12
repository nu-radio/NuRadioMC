from NuRadioReco.detector.ARIANNA import analog_components
import numpy as np
from NuRadioReco.utilities import units, fft
import time
import logging

logger = logging.getLogger("hardwareResponseIncorporator")


class hardwareResponseIncorporator:
    """
    Incorporates the gain and phase induced by the ARIANNA hardware.


    """

    def __init__(self):
        self.__debug = False
        self.__time_delays = {}
        self.__t = 0
        self.begin()

    def begin(self, debug=False):
        self.__debug = debug

    def run(self, evt, station, det, sim_to_data=False, zero_noise=False):
        """
        Switch sim_to_data to go from simulation to data or otherwise.
        The option zero_noise can be used to zero the noise around the pulse. It is unclear, how useful this is.
        """
        t = time.time()
        station_id = station.get_id()
        channels = station.get_channels()
        frequencies = channels[0].get_frequencies()  # get frequencies, the sampling rate is assumed to be the same for all channels
        # buffer amplifier and cable responses
#         cable_time_delay = self.__calculate_time_delays_cable()
#         logger.debug("cable time delay is {}ns".format(cable_time_delay / units.ns))

        for channel in channels:
            amp_type = det.get_amplifier_type(station_id, channel.get_id())
            trace_fft = channel.get_frequency_spectrum()
            amp_response = analog_components.get_amplifier_response(frequencies, amp_type=amp_type)
            cable_response = analog_components.get_cable_response_parametrized(frequencies, *det.get_cable_type_and_length(station.get_id(), channel.get_id()))
            if sim_to_data:
                if(self.__debug):
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(1, 1)
                    ax.plot(frequencies, amp_response['gain'], label='amp response {} series'.format(amp_type))
                    ax.plot(frequencies, cable_response / cable_response.max() * amp_response['gain'].max(), label='cable response')
                    ax.legend()
                trace_after_amp_fft = trace_fft * amp_response['gain'] * amp_response['phase']
                trace_after_cable_fft = trace_after_amp_fft * cable_response
                # zero first bins to avoid DC offset
                trace_after_cable_fft[0] = 0
                channel.set_frequency_spectrum(trace_after_cable_fft, channel.get_sampling_rate())

                # Cable delays, cable delays are differently treated for sim->data and deconvolution.
                # in case of simulations, the trace lenght is long enough that the trace is just rolled.
                # from the trigger simulator it will be cut to the correct length
                # in case of cable devoncolution, we don't roll the trace because of its limited size.
                # Instead we just save the cable delays in the trace start time
                roll_by = np.int(np.round((det.get_cable_delay(station_id, channel.get_id()) * channel.get_sampling_rate())))

#                 roll_by -= int(self.get_time_delay(amp_type) * channel.get_sampling_rate())

                trace = channel.get_trace()
                trace = np.roll(trace, roll_by)
                channel.set_trace(trace, channel.get_sampling_rate())

            else:
                trace_before_cable_fft = trace_fft / (cable_response)
                trace_before_amp_fft = trace_before_cable_fft / (amp_response['gain'] * amp_response['phase'])

                # Include cable delays
                cable_delay = det.get_cable_delay(station_id, channel.get_id())
                logger.debug("cable delay of channel {} is {}ns".format(channel.get_id(), cable_delay / units.ns))
                channel.add_trace_start_time(-cable_delay)
#                 roll_by = 0
#                 roll_by += int(self.get_time_delay(amp_type) * channel.get_sampling_rate())

                if self.__debug:
                    if(channel.get_id() in [0, 1, 2, 3]):
                        continue
                    import matplotlib.pyplot as plt
                    plt.figure()
                    plt.plot(channel.get_trace())
                    plt.title("Before rolling")

                    plt.figure()
                    plt.plot(channel.get_frequencies(), 1 / (np.abs(amp_response['gain'] * amp_response['phase'])))
                    plt.title("amp response")

                    plt.figure()
                    plt.plot(channel.get_frequencies(), np.abs(trace_fft), c='b', label='raw')
                    plt.plot(channel.get_frequencies(), np.abs(trace_before_cable_fft), c='r', label='including cable response')
#                     plt.figure()
                    plt.plot(channel.get_frequencies(), np.abs(trace_before_amp_fft), c='g', label='including amp response')
                    plt.show()

                # hardwareResponse incorporator should always be used in conjunction with bandpassfilter
                # otherwise, noise will be blown up
                channel.set_frequency_spectrum(trace_before_amp_fft, channel.get_sampling_rate())
#                 trace = channel.get_trace()
#                 trace = np.roll(trace, roll_by)

                if zero_noise:
                    if amp_type == '100':
                        t_0 = 10 * units.ns
                        t_1 = 140 * units.ns
                        signal_window = ((channel.get_times() > t_0) &
                                        (channel.get_times() < t_1))

                        trace[~signal_window] = 0

                        if self.__debug:
                            f1.axes[channel.get_id()].plot(channel.get_times(), trace)
                            f1.axes[channel.get_id()].axvspan(t_0, t_1, alpha=0.5, color='C5')
                    else:
                        logger.warning("Zero noise not implemented for amp other than 100 series")

#                 channel.set_trace(trace, channel.get_sampling_rate())
        self.__t += time.time() - t

    def end(self):
        from datetime import timedelta
        logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        logger.info("total time used by this module is {}".format(dt))
        return dt

    def __calculate_time_delays_cable(self):
        """
        helper function to calculate the time delay of the amp for a delta pulse
        """
        sampling_rate = 10 * units.GHz  # assume a huge sampling rate to have a good time resolution
        n = 2 ** 12
        trace = np.zeros(n)
        trace[n // 2] = 1
        max_time = trace.argmax() / sampling_rate
        spec = fft.time2freq(trace)
        ff = np.fft.rfftfreq(n, 1. / sampling_rate)
        response = analog_components.get_cable_response(ff)
        response_gain = response['gain']
        response_phase = response['phase']
        trace2 = fft.freq2time(spec * response_gain * response_phase)
#         import matplotlib.pyplot as plt
#         fig, ax = plt.subplots(1, 1)
#         ax.plot(trace)
#         ax.plot(trace2)
#         plt.show()
        max_time2 = np.abs(trace2).argmax() / sampling_rate
        return max_time2 - max_time

    def __calculate_time_delays_amp(self, amp_type):
        """
        helper function to calculate the time delay of the amp for a delta pulse
        """
        amp_response_f = analog_components.amplifier_response[amp_type]
        sampling_rate = 10 * units.GHz  # assume a huge sampling rate to have a good time resolution
        n = 2 ** 12
        trace = np.zeros(n)
        trace[n // 2] = 1
        max_time = trace.argmax() / sampling_rate
        spec = fft.time2freq(trace)
        ff = np.fft.rfftfreq(n, 1. / sampling_rate)
        amp_response_gain = amp_response_f['gain'](ff)
        amp_response_phase = amp_response_f['phase'](ff)
        mask = (ff < 70 * units.MHz) & (ff > 40 * units.MHz)
        spec[~mask] = 0
        trace2 = fft.freq2time(spec * amp_response_gain * amp_response_phase)
#         import matplotlib.pyplot as plt
#         fig, ax = plt.subplots(1, 1)
#         ax.plot(trace)
#         ax.plot(trace2)
#         plt.show()
        max_time2 = np.abs(trace2).argmax() / sampling_rate
        return max_time2 - max_time

    def get_time_delay(self, amp_type):
        if(amp_type not in self.__time_delays.keys()):
            # not yet calculated -> calculate the time delay
            self.__time_delays[amp_type] = self.__calculate_time_delays_amp(amp_type)
            logger.info("time delays of amp {} have not yet been calculated -> calculating -> time delay is {:.2f} ns".format(amp_type, self.__time_delays[amp_type] / units.ns))
        return self.__time_delays[amp_type]
