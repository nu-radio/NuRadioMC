from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import signal_processing
import numpy as np
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

        Channels with the same number of samples and the same current and target sampling rate
        are resampled together as one 2D array.

        Parameters
        ----------

        evt, station, det
            Event, Station, Detector
        sampling_rate: float (Default: None)
            In units '1 / time' provides the desired sampling rate of the data.
            If None, take sampling rate from detector description.
        """
        groups = {}
        for channel in station.iter_channels():
            target_sampling_rate = sampling_rate
            if target_sampling_rate is None:
                target_sampling_rate = det.get_sampling_frequency(station.get_id(), channel.get_id())

            if target_sampling_rate == channel.get_sampling_rate():
                continue

            key = (channel.get_number_of_samples(), channel.get_sampling_rate(), target_sampling_rate)
            groups.setdefault(key, []).append(channel)

        for (_, current_sampling_rate, target_sampling_rate), channels in groups.items():
            traces = np.stack([channel.get_trace() for channel in channels])
            resampled_traces = signal_processing.resample(traces, target_sampling_rate / current_sampling_rate)
            for channel, trace in zip(channels, resampled_traces):
                channel.set_trace(trace, target_sampling_rate)

    def end(self):
        pass
