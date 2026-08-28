from NuRadioReco.modules.base.module import register_run
import os
import NuRadioReco.framework.event
import NuRadioReco.framework.station
import NuRadioReco.framework.channel
import NuRadioReco.modules.io.eventWriter
from NuRadioReco.utilities import units
from NuRadioReco.utilities.LOFAR import NOISE_LIBRARY_DIRECTORY, NOISE_LIBRARY_NUR_FILEPATH
import numpy as np
import tempfile

import logging
logger = logging.getLogger('NuRadioReco.modules.measured_noise.LOFAR.noiseLibraryConverter')

class LOFARnoiseLibraryConverter:
    """
    Converts the real noise library for LOFAR from Karen Terveer and stores into a .nur file that can be used in the 
    channelMeasuredNoiseAdder (in the parent directory of this directory).

    The channelMeasuredNoiseAdder requires .nur files that contain the noise trace in (at least) a single station and needs
    only a single channel, accessed by noise_station.get_channel. 
    """
    def __init__(self):
        self.__filename = None
        self.__noise_trace = None
        self.__noise_sampling_rate = None
        self.__nur_filepath = None

        # some fixed values for the event ID
        # in principle doesnt need to be changed so set as a private member for now
        self.__noise_event_ID = 0

    def get_nur_filepath(self):
        """Return the nur filepath for usage in channelMeasuredNoiseAdder"""
        return self.__nur_filepath

    def begin(self, library_filename=NOISE_LIBRARY_DIRECTORY, nur_filename = NOISE_LIBRARY_NUR_FILEPATH, log_level = logging.INFO):
        """
        Loads in the noise library from the given filename.

        Parameters
        ----------
        filename : str, default=NOISE_LIBRARY_DIRECTORY
            the path to the noise library .npy file that consists of a single noise trace
            aggregated over different LOFAR events
        nur_filename : str, default=NOISE_LIBRARY_NUR_FILEPATH
            the path to the resulting .nur file that consists of the event trace
        log_level : logging object, default logging.INFO
            the level of the logger
        """
        logger.setLevel(log_level)
        
        # make sure it exists
        if not os.path.exists(library_filename):
            raise FileNotFoundError(f"File {library_filename} not found.")

        # also check if it has a .npy extension
        if library_filename.find(".npy") < 0:
            raise ValueError(f"File {library_filename} does not have a .npy extension.")

        # load using np.load
        self.__noise_trace = np.load(library_filename) * units.V
        self.__noise_sampling_rate = 200 * units.MHz # fixed here, since we know that timing resolution for LOFAR is 5 ns
        
        logger.info(f"Loaded noise trace from {library_filename} with sampling rate {self.__noise_sampling_rate / units.MHz} MHz")

        # set the nur file path
        # if None, then create a temporary directory which will be deleted at the end() function
        if nur_filename == None:
            temp_dir = tempfile.mkdtemp()
            self.__nur_filepath = os.path.join(temp_dir, "LOFAR_noise_trace.nur")
        else:
            self.__nur_filepath = nur_filename
        logger.info(f"Creating the noise .nur file into directory {self.__nur_filepath}")
        

    @register_run()
    def run(self, event, det):
        """
        Generates a .nur file that consists of a single event, consisting of a single station and channel
        which contains the noise trace.

        The event object is just a place holder, but the detector object is needed since we need to add the 
        channel ID corresponding to the same station & channel ID as the detector object used.

        Parameters
        ----------
        event: event object, just placeholders
        det : detector object, same as one used for the adder
        """
        # generate a single event object
        noise_evt = NuRadioReco.framework.event.Event(1, self.__noise_event_ID)

        # get a single station & channel IDs from the detector object
        noise_station_id  = det.get_station_ids()[0]
        noise_channel_id = det.get_channel_ids(noise_station_id)[0]

        # make a new station object
        noise_station = NuRadioReco.framework.station.Station(noise_station_id)
        
        # define a new channel and add the noise trace to it. 
        noise_channel = NuRadioReco.framework.channel.Channel(noise_channel_id)
        noise_channel.set_trace(self.__noise_trace, self.__noise_sampling_rate)
        noise_station.add_channel(noise_channel)
    
        # then add the station into the new event
        noise_evt.set_station(noise_station)
        logger.info("Added noise trace to all stations & channels.")

        # write using the event writer
        writer = NuRadioReco.modules.io.eventWriter.eventWriter()
        writer.begin(self.__nur_filepath)
        writer.run(noise_evt)
        writer.end()

        # now to appropriately map from any channel ID to the noise channel ID
        # we need to generate a mapping (dictionary) that will be passed
        # into channelMeasuredNoiseAdder
        channel_mapping = {}
        for station_id in det.get_station_ids():
            for channel_id in det.get_channel_ids(station_id):
                channel_mapping[channel_id] = noise_channel_id

        return channel_mapping
        

    def end(self):
        pass