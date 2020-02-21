from __future__ import absolute_import, division, print_function
from six import iteritems
try:
    import cPickle as pickle
except ImportError:
    import pickle


def deserialize(triggers_pkl):
    triggers = {}
    for data_pkl in triggers_pkl:
        trigger_type = pickle.loads(data_pkl)['_trigger_type']
        if(trigger_type == 'default'):
            trigger = Trigger(None)
            trigger.deserialize(data_pkl)
        elif(trigger_type == 'simple_threshold'):
            trigger = SimpleThresholdTrigger(None, None)
            trigger.deserialize(data_pkl)
        elif(trigger_type == 'high_low'):
            trigger = HighLowTrigger(None, None, None, None, None)
            trigger.deserialize(data_pkl)
        elif(trigger_type == 'simple_phased'):
            trigger = SimplePhasedTrigger(None, None)
            trigger.deserialize(data_pkl)
        elif(trigger_type == 'envelope_trigger'):
            trigger = EnvelopeTrigger(None, None, None, None)
            trigger.deserialize(data_pkl)
        else:
            raise ValueError("unknown trigger type")
        triggers[trigger.get_name()] = trigger
    return triggers


class Trigger:
    """
    base class to store different triggers
    """

    def __init__(self, name, channels=None, trigger_type='default'):
        """
        initialize trigger class

        Parameters
        -----------
        name: string
            unique name of the trigger
        channels: array of ints
            the channels that are involved in the trigger
        type: string
            the trigger type
        """
        self._name = name
        self._channels = channels
        self._triggered = False
        self._trigger_time = None
        self._trigger_times = None
        self._trigger_type = trigger_type
        self._triggered_channels = []

    def has_triggered(self):
        """
        returns true if station has trigger, false otherwise
        """
        return self._triggered

    def set_triggered(self, triggered=True):
        """ set the trigger to True or False """
        self._triggered = triggered

    def set_trigger_time(self, time):
        """ set the trigger time

        Parameters
        ----------
        time: float
            the trigger time from the beginning of the trace
        """
        self._trigger_time = time

    def get_trigger_time(self):
        """
        get the trigger time (time with respect to beginning of trace)
        """
        return self._trigger_time

    def set_trigger_times(self, times):
        """ set the trigger times

        Parameters
        ----------
        times: array
            all trigger times
        """
        self._trigger_times = times

    def get_trigger_times(self):
        """
        get the trigger times (time with respect to beginning of trace)
        """
        return self._trigger_times

    def get_name(self):
        """ get trigger name """
        return self._name

    def get_type(self):
        """ get trigger type """
        return self._type

    def get_triggered_channels(self):
        """ get IDs of channels that have triggered """
        return self._triggered_channels

    def set_triggered_channels(self, triggered_channels):
        self._triggered_channels = triggered_channels

    def serialize(self):
        return pickle.dumps(self.__dict__, protocol=4)

    def deserialize(self, data_pkl):
        for key, value in iteritems(pickle.loads(data_pkl)):
            setattr(self, key, value)

    def __str__(self):
        output = ""
        for key, value in iteritems(self.__dict__):
            output += "{}: {}\n".format(key[1:], value)
        return output

    def get_trigger_settings(self):
        output = {}
        for key, value in iteritems(self.__dict__):
            output[key[1:]] = value
        return output


class SimpleThresholdTrigger(Trigger):

    def __init__(self, name, threshold, channels=None, number_of_coincidences=1,
                 channel_coincidence_window=None):
        """
        initialize trigger class

        Parameters
        -----------
        name: string
            unique name of the trigger
        threshold: float
            the threshold
        channels: array of ints or None
            the channels that are involved in the trigger
            default: None, i.e. all channels
        number_of_coincidences: int
            the number of channels that need to fulfill the trigger condition
            default: 1
        channel_coincidence_window: float or None (default)
            the coincidence time between triggers of different channels
        """
        Trigger.__init__(self, name, channels, 'simple_threshold')
        self._threshold = threshold
        self._number_of_coincidences = number_of_coincidences
        self._coinc_window = channel_coincidence_window


class EnvelopePhasedTrigger(Trigger):

    def __init__(self, name, threshold_factor, power_mean, power_std,
                 triggered_channels=None, phasing_angles=None, trigger_delays=None,
                 output_passband=(None, None)):
        """
        initialize trigger class
        Parameters
        -----------
        name: string
            unique name of the trigger
        threshold_factor: float
            the threshold factor
        power_mean: float
            mean of the noise trace after being filtered with the diode
        power_std: float
            standard deviation of the noise trace after being filtered with the
            diode. power_mean and power_std can be calculated with the function
            calculate_noise_parameters from utilities.diodeSimulator
        triggered_channels: array of ints or None
            the channels that are involved in the main phased beam
            default: None, i.e. all channels
        phasing_angles: array of floats or None
            the angles for each beam
        trigger_delays: dictionary
            the delays for the channels that have caused a trigger.
            If there is no trigger, it's an empty dictionary
        output_passband: (float, float) tuple
            Frequencies for a 6th-order Butterworth filter to be applied after
            the diode filtering.
        """
        Trigger.__init__(self, name, triggered_channels, 'envelope_phased')
        self._triggered_channels = triggered_channels
        self._phasing_angles = phasing_angles
        self._threshold_factor = threshold_factor
        self._power_mean = power_mean
        self._power_std = power_std
        self._trigger_delays = trigger_delays
        self._output_passband = output_passband


class SimplePhasedTrigger(Trigger):

    def __init__(self, name, threshold, channels=None, secondary_channels=None,
                 primary_angles=None, secondary_angles=None,
                 trigger_delays=None, sec_trigger_delays=None):
        """
        initialize trigger class
        Parameters
        -----------
        name: string
            unique name of the trigger
        threshold: float
            the threshold
        channels: array of ints or None
            the channels that are involved in the main phased beam
            default: None, i.e. all channels
        secondary_channels: array of ints or None
            the channels involved in the secondary phased beam
        primary_angles: array of floats or None
            the angles for each subbeam of the primary phasing
        secondary_angles: array of floats or None
            the angles for each subbeam of the secondary phasing
        trigger_delays: dictionary
            the delays for the primary channels that have caused a trigger.
            If there is no trigger, it's an empty dictionary
        sec_trigger_delays: dictionary
            the delays for the secondary channels that have caused a trigger.
            If there is no trigger or no secondary channels, it's an empty dictionary
        """
        Trigger.__init__(self, name, channels, 'simple_phased')
        self._primary_channels = channels
        self._primary_angles = primary_angles
        self._secondary_channels = secondary_channels
        self._secondary_angles = secondary_angles
        self._threshold = threshold
        self._trigger_delays = trigger_delays
        self._sec_trigger_delays = sec_trigger_delays


class HighLowTrigger(Trigger):

    def __init__(self, name, threshold_high, threshold_low, high_low_window,
                 channel_coincidence_window, channels=None, number_of_coincidences=1):
        """
        initialize trigger class

        Parameters
        -----------
        name: string
            unique name of the trigger
        threshold_high: float
            the high threshold
        threshold_low: float
            the low threshold
        high_low_window: float
            the coincidence time between a high and low per channel
        channel_coincidence_window: float
            the coincidence time between triggers of different channels
        channels: array of ints or None
            the channels that are involved in the trigger
            default: None, i.e. all channels
        number_of_coincidences: int
            the number of channels that need to fulfill the trigger condition
            default: 1
        """
        Trigger.__init__(self, name, channels, 'high_low')
        self._number_of_coincidences = number_of_coincidences
        self._threshold_high = threshold_high
        self._threshold_low = threshold_low
        self._high_low_window = high_low_window
        self._coinc_window = channel_coincidence_window


class IntegratedPowerTrigger(Trigger):

    def __init__(self, name, threshold, channel_coincidence_window, channels=None, number_of_coincidences=1,
                 power_mean=None, power_std=None, integration_window=None):
        """
        initialize trigger class

        Parameters
        -----------
        name: string
            unique name of the trigger
        threshold: float
            the threshold
        channel_coincidence_window: float
            the coincidence time between triggers of different channels
        channels: array of ints or None
            the channels that are involved in the trigger
            default: None, i.e. all channels
        number_of_coincidences: int
            the number of channels that need to fulfill the trigger condition
            default: 1
        """
        Trigger.__init__(self, name, channels, 'int_power')
        self._number_of_coincidences = number_of_coincidences
        self._threshold = threshold
        self._coinc_window = channel_coincidence_window
        self._power_mean = power_mean
        self._power_std = power_std
        self._integration_window = integration_window


class EnvelopeTrigger(Trigger):

    def __init__(self, name, passband, order, threshold, number_of_coincidences=2,
                 channel_coincidence_window=None, channels=None):
        """
        initialize trigger class

        Parameters
        -----------
        name: string
            unique name of the trigger
        passband: array
            the passband in which the trigger should apply
        order: int
            order of filtertype 'butterabs'
        threshold: float
            the threshold
        channels: array of ints or None
            the channels that are involved in the trigger
            default: None, i.e. all channels
        number_of_coincidences: int
            the number of channels that need to fulfill the trigger condition
            default: 1
        channel_coincidence_window: float or None (default)
            the coincidence time between triggers of different channels
        """
        Trigger.__init__(self, name, channels, 'envelope_trigger')
        self._passband = passband
        self._order = order
        self._threshold = threshold
        self._number_of_coincidences = number_of_coincidences
        self._coinc_window = channel_coincidence_window
