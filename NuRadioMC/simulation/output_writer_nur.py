import numpy as np
import NuRadioReco.modules.io.eventWriter
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.electricFieldResampler

class outputWriterNur:
    """
    Class to write simulation results into a .nur file
    """
    def __init__(
            self,
            output_filename,
            save_detector,
            detector
    ):
        """
        Initialize class

        Parameters
        ----------
        output_filename: string
            Name of the file the simulation results are written in.
        save_detector: boolean
            Specifies if the detector description is written into the .nur file
        detector: NuRadioReco.detector.detector.Detector class
            The detector description used in the simulation
        """
        self.__output_filename = output_filename
        self.__save_detector = save_detector
        self.__detector = detector
        self.__event_writer = NuRadioReco.modules.io.eventWriter.eventWriter()
        self.__event_writer.begin(
            self.__output_filename
        )
        self.__channel_resampler = NuRadioReco.modules.channelResampler.channelResampler()
        self.__electric_field_resampler = NuRadioReco.modules.electricFieldResampler.electricFieldResampler()
    
    def save_event(
        self,
        events
    ):
        """
        Save an event

        Parameters
        ----------
        events: dict of NuRadioReco.framework.event.Event objects
            A dictionary containint the events that are saved
        """
        for key in events:
            event = events[key]
            for station in event.get_stations():
                target_sampling_rate = self.__detector.get_sampling_frequency(
                    station.get_id(),
                    station.get_channel_ids()[0]
                )
                self.__channel_resampler.run(
                    event,
                    station,
                    self.__detector,
                    target_sampling_rate
                )
                self.__channel_resampler.run(
                    event,
                    station.get_sim_station(),
                    self.__detector,
                    target_sampling_rate
                )
                self.__electric_field_resampler.run(
                    event,
                    station.get_sim_station(),
                    self.__detector,
                    target_sampling_rate
                )
            if self.__save_detector:
                self.__event_writer.run(
                    event,
                    self.__detector
                )
            else:
                self.__event_writer.run(
                    event
                )
    
    def end(self):
        self.__event_writer.end()
