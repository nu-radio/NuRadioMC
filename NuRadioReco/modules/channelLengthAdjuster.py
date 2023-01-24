from NuRadioReco.modules.base.module import register_run
import numpy as np
import logging
logger = logging.getLogger('channelLengthAdjuster')


class channelLengthAdjuster:
    """
    cuts the trace to detector specifications, uses a simple algorithm to determine the location of the pulse
    """
    def __init__(self):
        self.number_of_samples = None
        self.offset = None
        self.begin(())

    def begin(self, number_of_samples=256, offset=50):
        """
        Defines number of samples to cut the data to and how many samples before maximum in trace

        Parameters
        ----------
        number_of_samples: int
            Number of samples desired in signal
        offset: int
            (roughly) How many samples before pulse

        """

        self.number_of_samples = number_of_samples
        self.offset = offset

    @register_run()
    def run(self, evt, station, det=None, channel_ids=None):
        """
        Run method for the channelLengthAdjuster

        Parameters
        ----------
        evt: Event object
            The event containing the station whose channels should be cut
        station: Station object
            The station whose channels should be cut
        det: Detector object
            The detector description. It is not actually needed and just
            kept as a parameter to comply with the module template
        channel_ids: List of ints
            Can be used to specify, which channels shall be cut. If it is None,
            all channels are cut.
        """
        max_pos = []
        if channel_ids is None:
            for channel in station.iter_channels():
                max_pos.append(np.argmax(np.abs(channel.get_trace())))
        else:
            for channel_id in channel_ids:
                channel = station.get_channel(channel_id)
                max_pos.append(np.argmax(np.abs(channel.get_trace())))

        pulse_start = min(max_pos) - self.offset
        for channel in station.iter_channels():
            trace = channel.get_trace()
            if self.number_of_samples > trace.shape[0]:
                logger.warning("Input has fewer samples than desired output. Channels has only {} samples but {} samples are requested.".format(trace.shape[0], self.number_of_samples))
                new_trace = np.zeros(self.number_of_samples)
                new_trace[:trace.shape[0]] = trace
            elif self.number_of_samples == trace.shape[0]:
                logger.warning("Channel already at desired length, nothing done.")
                continue
            else:
                # shift trace to be in the correct location for cutting
                new_trace = np.roll(trace, -1 * pulse_start)
                new_trace = new_trace[:self.number_of_samples]
                channel.set_trace_start_time(channel.get_trace_start_time() + pulse_start / channel.get_sampling_rate())

            channel.set_trace(new_trace, channel.get_sampling_rate())

    def end(self):
        pass
