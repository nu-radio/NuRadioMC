"""
This module implements a `Channel` similar to the `NuRadioReco.framework.channel.Channel` object.

It is however intended to be used for non-radio channels, e.g. particle detector channels.

"""

import NuRadioReco.framework.channel
from NuRadioReco.framework import parameters
import NuRadioReco.framework.parameter_serialization
try:
    import cPickle as pickle
except ImportError:
    import pickle

import logging
logger = logging.getLogger('NuRadioReco.hybrid_channel')



class Channel:

    def __init__(self, channel_id, channel_group_id=None):
        self.__channel = NuRadioReco.framework.channel.Channel(channel_id, channel_group_id)

    def get_parameter(self, key):
        if not isinstance(key, parameters.hybridChannelParameters):
            logger.error("parameter key needs to be of type NuRadioReco.framework.parameters.hybridChannelParameters")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.hybridChannelParameters")
        return self._parameters[key]

    def get_parameters(self):
        return self._parameters

    def set_parameter(self, key, value):
        if not isinstance(key, parameters.hybridChannelParameters):
            logger.error("parameter key needs to be of type NuRadioReco.framework.parameters.hybridChannelParameters")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.hybridChannelParameters")
        self._parameters[key] = value

    def has_parameter(self, key):
        if not isinstance(key, parameters.hybridChannelParameters):
            logger.error("parameter key needs to be of type NuRadioReco.framework.parameters.hybridChannelParameters")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.hybridChannelParameters")
        return key in self._parameters


    def __setitem__(self, key, value):
        self.set_parameter(key, value)

    def __getitem__(self, key):
        return self.get_parameter(key)

    ## We 'partially inherit' from the NuRadioReco.framework.channel.Channel class
    ## That is, internally a particle channel is the same as a radio channel,
    ## but we expose only a limited subset of the methods (e.g. frequency spectra don't make sense)

    def get_id(self):
        return self.__channel.get_id()

    def get_group_id(self):
        """
        channel group id

        If no group id is specified, the channel id is returned. This allows using modules that use the `group_id`
        feature also on detector setups that don't use this feature.
        """
        return self.__channel.get_group_id()


    def get_trace(self):
        """
        Returns the time trace.

        If the frequency spectrum was modified before,
        an ifft is performed automatically to have the time domain representation
        up to date.

        Returns
        -------
        trace: np.array of floats
            the time trace
        """
        return self.__channel.get_trace()

    def set_trace(self, trace, sampling_rate):
        """
        Sets the time trace.

        Parameters
        ----------
        trace : np.array of floats
            The time series
        sampling_rate : float or str
            The sampling rate of the trace, i.e., the inverse of the bin width.
            If `sampling_rate="same"`, sampling rate is not changed (requires previous initialisation).
        """
        self.__channel.set_trace(trace=trace, sampling_rate=sampling_rate)


    def get_sampling_rate(self):
        """
        Returns the sampling rate of the trace.

        Returns
        -------
        sampling_rate: float
            sampling rate, i.e., the inverse of the bin width
        """
        return self.__channel.get_sampling_rate()

    def get_times(self):
        return self.__channel.get_times()

    def set_trace_start_time(self, start_time):
        self.__channel.set_trace_start_time(start_time=start_time)

    def add_trace_start_time(self, start_time):
        self.__channel.add_trace_start_time(start_time=start_time)

    def get_trace_start_time(self):
        return self.__channel.get_trace_start_time()

    def get_number_of_samples(self):
        """
        Returns the number of samples in the time domain.

        Returns
        -------
        n_samples: int
            number of samples in time domain
        """
        self.__channel.get_number_of_samples()


    def serialize(self, save_trace=False):
        data = {
            'parameters': NuRadioReco.framework.parameter_serialization.serialize(self._parameters),
            'channel': self.__channel.serialize(save_trace=save_trace),
        }

        return pickle.dumps(data, protocol=4)

    def deserialize(self, data_pkl):
        data = pickle.loads(data_pkl)
        self._parameters = NuRadioReco.framework.parameter_serialization.deserialize(data['parameters'], parameters.channelParameters)
        self.__channel.deserialize(data['channel'])