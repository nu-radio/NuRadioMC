from NuRadioReco.modules.base.module import register_run
import numpy.random


class channelTimeJitterSimulator:
    """
    Module to simulate small time jitters in a waveform by applying random time shifts.
    """
    def begin(self):
        pass

    @register_run()
    def run(self, event, station, det, time_shift):
        """
        Apply random time shifts to all channel waveforms of the station

        Parameters
        --------------------
        event: Event object
            The event containing the stations the time shift should be applied to.
        station: Station object
            The station to whose channels the time shift is applied.
        det: Detector object
            The detector description
        time_shift: number
            The scale of the time shifts that should be applied. For each channel, the applied time shift is randomly
            drawn from a normal distribution centered around 0 with standard deviatoin equal to time_shift.
        """
        for channel in station.iter_channels():
            channel.apply_time_shift(numpy.random.normal(0, time_shift), True)

    def end(self):
        pass
