from NuRadioReco.modules.base.module import register_run
import NuRadioReco.modules.io.NuRadioRecoio
import numpy as np
from NuRadioReco.utilities import units
import numpy.random
import logging
import matplotlib.pyplot as plt


class channelMeasuredNoiseAdder:
    """
    Module that adds measured noise to channel traces
    It does so by reading in a set of .nur files, randomly selecting a forced trigger event
    and adding the noise from it to the channel waveforms.
    The waveforms from the channels in the noise files need to be at least as long as the
    waveforms to which the noise is added, so it is recommended to cut them to the right size
    first, for example using the channelLengthAdjuster.
    """
    def __init__(self):
        self.__filenames = None
        self.__io = None
        self.__random_state = None
        self.__max_iterations = None
        self.__debug = None
        self.logger = logging.getLogger('NuRadioReco.channelMeasuredNoiseAdder')
        self.__noise_data = None

    def begin(self, filenames, random_seed=None, max_iterations=100, debug=False, draw_noise_statistics=False):
        """
        Set up module parameters

        Parameters
        ----------
        filenames: list of strings
            List of .nur files containing the measured noise
        random_seed: int, default: None
            Seed for the random number generator. By default, no seed is set.
        max_iterations: int, default: 100
            The module will pick a random event from the noise files, until a suitable event is found
            or until the number of iterations exceeds max_iterations. In that case, an error is thrown.
        debug: bool, default: False
            Set True to get debug output
        draw_noise_statistics: boolean, default: False
            If true, the values of all samples is stored and a histogram with noise statistics is drawn
            be the end() method
        """
        self.__filenames = filenames
        self.__io = NuRadioReco.modules.io.NuRadioRecoio.NuRadioRecoio(self.__filenames)
        self.__random_state = numpy.random.Generator(numpy.random.Philox(random_seed))
        self.__max_iterations = max_iterations
        if debug:
            self.logger.setLevel(logging.DEBUG)
            self.logger.debug('Reading noise from {} files containing {} events'.format(len(filenames), self.__io.get_n_events()))
        if draw_noise_statistics:
            self.__noise_data = []

    @register_run()
    def run(self, event, station, det):
        """
        Add measured noise to station channels

        Parameters
        ----------
        event: event object
        station: station object
        det: detector description
        """
        noise_station = None
        for i in range(self.__max_iterations):
            noise_station = self.get_noise_station(station)
            # Get random station from noise file. If we got a suitable station, we continue,
            # otherwise we try again
            if noise_station is not None:
                break
        # To avoid infinite loops, if no suitable noise station was found after a number of iterations we raise an error
        if noise_station is None:
            raise ValueError('Could not find suitable noise event in noise files after {} iterations'.format(self.__max_iterations))
        for channel in station.iter_channels():
            noise_channel = noise_station.get_channel(channel.get_id())
            channel_trace = channel.get_trace()
            if noise_channel.get_sampling_rate() != channel.get_sampling_rate():
                noise_channel.resample(channel.get_sampling_rate())
            noise_trace = noise_channel.get_trace()
            channel_trace += noise_trace[:channel.get_number_of_samples()]
            if self.__noise_data is not None:
                self.__noise_data.append(noise_trace)

    def get_noise_station(self, station):
        """
        Returns a random station from the noise files that can be used as a noise sample.
        The function selects a random event from the noise files and checks if it is suitable.
        If it is, the station is returned, otherwise None is returned. The event is suitable if it
        fulfills these criteria:
    
        * It contains a station with the same station ID as the one to which the noise shall be added
        * The station does not have a trigger that has triggered.
        * The every channel in the station to which the noise shall be added is also present in the station

        Parameters
        ----------
        station: Station class
            The station to which the noise shall be added
        """
        event_i = self.__random_state.randint(self.__io.get_n_events())
        noise_event = self.__io.get_event_i(event_i)
        if station.get_id() not in noise_event.get_station_ids():
            self.logger.debug('No station with ID {} found in event'.format(station.get_id()))
            return None
        noise_station = noise_event.get_station(station.get_id())
        for trigger_name in noise_station.get_triggers():
            trigger = noise_station.get_trigger(trigger_name)
            if trigger.has_triggered():
                self.logger.debug('Noise station has triggered')
                return None
        for channel_id in station.get_channel_ids():
            if channel_id not in noise_station.get_channel_ids():
                self.logger.debug('Channel {} found in station but not in noise file'.format(channel_id))
                return None
            noise_channel = noise_station.get_channel(channel_id)
            channel = station.get_channel(channel_id)
            if noise_channel.get_number_of_samples() / noise_channel.get_sampling_rate() < channel.get_number_of_samples() / channel.get_sampling_rate():
                return None
        return noise_station

    def end(self):
        """
        End method. Draws a histogram of the noise statistics and fits a
        Gaussian distribution to it.
        """
        if self.__noise_data is not None:
            noise_entries = np.array(self.__noise_data)
            noise_bins = np.arange(-150, 150, 5.) * units.mV
            noise_entries = noise_entries.flatten()
            mean = noise_entries.mean()
            sigma = np.sqrt(np.mean((noise_entries - mean)**2))
            plt.close('all')
            fig1 = plt.figure()
            ax1_1 = fig1.add_subplot(111)
            n, bins, pathes = ax1_1.hist(noise_entries / units.mV, bins=noise_bins / units.mV)
            ax1_1.plot(noise_bins / units.mV, np.max(n) * np.exp(-.5 * (noise_bins - mean)**2 / sigma**2))
            ax1_1.grid()
            ax1_1.set_xlabel('sample value [mV]')
            ax1_1.set_ylabel('entries')
            plt.show()
