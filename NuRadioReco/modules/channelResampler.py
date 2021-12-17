from NuRadioReco.modules.base.module import register_run
import logging


class channelResampler:
    """
    Resamples the trace to a new sampling rate.
    """

    def __init__(self):
        self.logger = logging.getLogger('NuRadioReco.channelResampler')
        self.__debug = None
        self.__max_upsampling_factor = None
        self.begin()

    def begin(self, debug=False, log_level=logging.WARNING):
        self.__max_upsampling_factor = 5000
        self.__debug = debug
        self.logger.setLevel(log_level)

        """
        Begin the channelResampler

        Parameters
        ----------

        __debug: bool
            Debug switch

        """

    @register_run()
    def run(self, evt, station, det, sampling_rate):
        """
        Run the channelResampler

        Parameters
        ----------

        evt, station, det
            Event, Station, Detector
        sampling_rate: float
            In units 1/time provides the desired sampling rate of the data.

        """
        for channel in station.iter_channels():
            channel.resample(sampling_rate)

    def end(self):
        pass
