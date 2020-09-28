from NuRadioReco.modules.base.module import register_run
import logging
logger = logging.getLogger('stationResampler')


class electricFieldResampler:
    """
    resamples the electric field trace to a new sampling rate
    """

    def __init__(self):
        self.__max_upsampling_factor = None
        self.begin()

    def begin(self):
        self.__max_upsampling_factor = 5000
        pass

    @register_run()
    def run(self, event, station, det, sampling_rate):
        """
        resample electric field

        Parameters
        ----------
        event: event

        station: station

        det: detector

        sampling_rate: float
            desired new sampling rate

        """
        for efield in station.get_electric_fields():
            efield.resample(sampling_rate)

    def end(self):
        pass
