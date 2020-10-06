from NuRadioReco.modules.base.module import register_run
import NuRadioReco.modules.io.NuRadioRecoio
import numpy as np
from NuRadioReco.utilities import units, fft
import numpy.random
import logging
import matplotlib.pyplot as plt


class channelMeasuredNoiseAdder:
    def __init__(self):
        self.__filenames = None
        self.__io = None
        self.__random_state = None
        self.__max_iterations = None
        self.__debug = None
        self.logger = logging.getLogger('NuRadioReco.channelMeasuredNoiseAdder')
        self.__noise_data = []

    def begin(self, filenames, random_seed=None, max_iterations=100, debug=False):
        self.__filenames = filenames
        self.__io = NuRadioReco.modules.io.NuRadioRecoio.NuRadioRecoio(self.__filenames)
        self.__random_state = numpy.random.RandomState(random_seed)
        self.__max_iterations = max_iterations
        if debug:
            self.logger.setLevel(logging.DEBUG)
            self.logger.debug('Reading noise from {} files containing {} events'.format(len(filenames), self.__io.get_n_events()))

    @register_run()
    def run(self, event, station, det):
        noise_station = None
        for i in range(self.__max_iterations):
            noise_station = self.get_noise_station(station)
            if noise_station is not None:
                break
        if noise_station is None:
            raise ValueError('Could not find suitable noise event in noise files after {} iterations'.format(self.__max_iterations))
        for channel in station.iter_channels():
            noise_channel = noise_station.get_channel(channel.get_id())
            channel_trace = channel.get_trace()
            noise_trace = noise_channel.get_trace()
            if channel.get_number_of_samples() > noise_channel.get_number_of_samples():
                self.logger.warning(
                    'Channel has more samples ({}) than noise channel ({})'.format(
                        channel.get_number_of_samples(),
                        noise_channel.get_number_of_samples()
                    )
                )
                channel_trace[:noise_channel.get_number_of_samples()] += noise_trace
            else:
                channel_trace += noise_trace[:channel.get_number_of_samples()]
            self.__noise_data.append(noise_trace)

    def get_noise_station(self, station):
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
        return noise_station

    def end(self):
        noise_entries = np.array(self.__noise_data)
        noise_bins = np.arange(-150, 150, 5.) * units.mV
        noise_entries = noise_entries.flatten()
        mean = noise_entries.mean()
        sigma = np.sqrt(np.mean((noise_entries - mean)**2))
        plt.close('all')
        fig1 = plt.figure()
        ax1_1 = fig1.add_subplot(111)
        n, bins, pathes = ax1_1.hist(noise_entries, bins=noise_bins)
        ax1_1.plot(noise_bins, np.max(n) * np.exp(-.5 * (noise_bins - mean)**2 / sigma**2))
        ax1_1.grid()
        plt.show()



