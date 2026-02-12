from NuRadioReco.modules.base.module import register_run
import numpy.random


class channelTimeJitterSimulator:
    def begin(self):
        pass

    @register_run()
    def run(self, event, station, det, time_shift):
        for channel in station.iter_channels():
            channel.apply_time_shift(numpy.random.normal(0, time_shift), True)

    def end(self):
        pass
