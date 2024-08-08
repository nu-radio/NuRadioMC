from NuRadioReco.modules.base.module import register_run
import logging


class channelResampler:
    """
    Resamples the trace to a new sampling rate.
    """

    def __init__(self):
        pass

    def begin(self):
        pass

    @register_run()
    def run(self, evt, station, det, sampling_rate=None):
        """
        Resample channel traces.

        Parameters
        ----------

        evt, station, det
            Event, Station, Detector
        sampling_rate: float (Default: None)
            In units '1 / time' provides the desired sampling rate of the data.
            If None, take sampling rate from detector description.
        """
        for channel in station.iter_channels():
            if sampling_rate is None:
                channel.resample(det.get_sampling_frequency(station.get_id(), channel.get_id()))
            else:
                channel.resample(sampling_rate)

    def end(self):
        pass
