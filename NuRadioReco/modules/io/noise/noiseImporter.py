from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units
import numpy as np
import logging
import glob
import os
import sys
from NuRadioReco.modules.io import NuRadioRecoio
logger = logging.getLogger('noiseImporter')


class noiseImporter:
    """
    Imports recorded noise from ARIANNA station. The recorded noise needs to match the station geometry and sampling
    as chosen with channelResampler and channelLengthAdjuster

    For different stations, new noise files need to be used.
    Collect forced triggers from any type of station to use for analysis.
    A seizable fraction of data is recommended for accuracy.

    The noise will be random. This module therefore might produce non-reproducible results on a single event basis,
    if run several times.
    """

    def __init__(self):
        self.__channel_mapping = None
        self.__station_id = None

    def begin(self, noise_folder, station_id=None, noise_files=None,
              channel_mapping=None, log_level=logging.WARNING, mean_opt=True):
        """
        Parameters
        ----------
        noise_folder: string
            the folder containing the noise files
        station_id: int
            the station id, specifies from which station the forced triggers are used
            as a noise sample. The data must have the naming convention 'forced_station_??.nur'
            where ?? is replaced with the station id. If station_id is None, the noiseImporter
            will try to find the station with the same iD as the one passed to the run
            function in the noise file
        channel_mapping: dict or None
            option relevant for MC studies of new station designs where we do not
            have forced triggers for. The channel_mapping dictionary maps the channel
            ids of the MC station to the channel ids of the noise data
            Default is None which is 1-to-1 mapping
        log_level: loggging log level
            the log level, default logging.WARNING
        mean_opt: boolean
            option to subtract mean from trace. Set mean_opt=False to remove mean subtraction from trace.
        """
        logger.setLevel(log_level)
        self.__channel_mapping = channel_mapping
        self.__mean_opt = mean_opt
        if(noise_files is not None):
            self.__noise_files = noise_files
        else:
            if(station_id is None):
                logger.error("noise_files and station_id can't be both None")
                sys.exit(-1)
            else:
                self.__noise_files = glob.glob(os.path.join(noise_folder, "forced_station_{}.nur".format(station_id)))
                if(len(self.__noise_files) == 0):
                    logger.error("no noise files found for station {} in folder {}".format(station_id, noise_folder))
                    sys.exit(-1)

        # open files and scan for number of events
        self.__open_files = []
        self.__n_tot = 0
        for file_name in self.__noise_files:
            f = NuRadioRecoio.NuRadioRecoio(file_name, parse_header=False)
            n = f.get_n_events()
            self.__open_files.append({'f': f, 'n_low': self.__n_tot,
                                      'n_high': self.__n_tot + n - 1})
            self.__n_tot += n

    def __get_noise_event(self, i):
        for f in self.__open_files:
            if(i >= f['n_low'] and i <= f['n_high']):
                return f['f'].get_event_i(i - f['n_low'])

    def __get_noise_channel(self, channel_id):
        if(self.__channel_mapping is None):
            return channel_id
        else:
            return self.__channel_mapping[channel_id]

    @register_run()
    def run(self, evt, station, det):
        # loop over stations in simulation
        i_noise = np.random.randint(0, self.__n_tot)
        noise_event = self.__get_noise_event(i_noise)
        if self.__station_id is None:
            station_id = station.get_id()
        else:
            station_id = self.__station_id
        noise_station = noise_event.get_station(station_id)
        if noise_station is None:
            raise KeyError('Station whith ID {} not found in noise file'.format(station_id))
        logger.info("choosing noise event {} ({}, run {}, event {}) randomly".format(i_noise, noise_station.get_station_time(),
                                                                                     noise_event.get_run_number(),
                                                                                     noise_event.get_id()))

        for channel in station.iter_channels():
            channel_id = channel.get_id()

            trace = channel.get_trace()
            noise_channel = noise_station.get_channel(self.__get_noise_channel(channel_id))
            noise_trace = noise_channel.get_trace()
            # check if trace has the same size
            if (len(trace) != len(noise_trace)):
                logger.error("Mismatch: Noise has {0} and simulation {1} samples".format(len(noise_trace), len(trace)))
                sys.exit(-1)
            # check sampling rate
            if (channel.get_sampling_rate() != noise_channel.get_sampling_rate()):
                logger.error("Mismatch in sampling rate: Noise has {0} and simulation {1} GHz".format(
                    noise_channel.get_sampling_rate() / units.GHz, channel.get_sampling_rate() / units.GHz))
                sys.exit(-1)

            trace = trace + noise_trace

            if self.__mean_opt:
                mean = noise_trace.mean()
                std = noise_trace.std()
                if(mean > 0.05 * std):
                    logger.warning(
                        "the noise trace has an offset of {:.2}mV which is more than 5\% of the STD of {:.2f}mV. The module corrects for the offset but it might points to an error in the FPN subtraction.".format(mean, std))
                trace = trace - mean

            channel.set_trace(trace, channel.get_sampling_rate())

    def end(self):
        for f in self.__open_files:
            f['f'].close_file()
