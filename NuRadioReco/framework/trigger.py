"""
Trigger classes: the result and configuration of a trigger, stored in a `Station` (see `BaseStation.get_trigger`).

All values of a trigger except its name and type are stored as `triggerParameters`
(``trigger[triggerParameters.threshold]``); the getters / setters of the classes are wrappers around them.
`deserialize` creates the class registered for the trigger type string.
"""
import logging
import pickle

import numpy as np

from NuRadioReco.framework.parameter_storage import ParameterStorage
from NuRadioReco.framework.parameters import triggerParameters as trp
from NuRadioReco.utilities.io_utilities import _dumps
from NuRadioReco.utilities import units

logger = logging.getLogger('NuRadioReco.framework.trigger')

_TRIGGER_CLASSES = {}  # trigger type string -> class, filled by `Trigger.__init_subclass__`


def deserialize(triggers_pkl):
    """
    Create the triggers from their serialized (pickled) form, see `Trigger.serialize`.

    Parameters
    ----------
    triggers_pkl : list of bytes
        The serialized triggers.

    Returns
    -------
    triggers : dict
        Trigger name -> trigger object (of the class registered for its type; unknown types
        are read as `Trigger`).
    """
    triggers = {}
    for data_pkl in triggers_pkl:
        data = pickle.loads(data_pkl)
        trigger_type = data['_trigger_type']
        cls = _TRIGGER_CLASSES.get(trigger_type)
        if cls is None:
            logger.warning(f"Unknown trigger type '{trigger_type}': read as base class `Trigger`.")
            cls = Trigger
        trigger = cls.__new__(cls)
        Trigger.__init__(trigger, None, trigger_type=trigger_type)  # defaults for values missing in legacy files
        trigger._set_state(data)
        triggers[trigger.get_name()] = trigger

    return triggers


# Attributes of triggers in .nur files written before the triggerParameters were introduced
# (the pickled ``__dict__`` of the trigger) -> parameter. Special cases are handled in `Trigger._set_legacy_state`.
_LEGACY_ATTRIBUTES = {
    '_channels': trp.channels,
    '_triggered': trp.triggered,
    '_trigger_time': trp.trigger_time,
    '_trigger_times': trp.trigger_times,
    '_pre_trigger_times': trp.pre_trigger_times,
    '_threshold': trp.threshold,
    '_threshold_high': trp.threshold_high,
    '_threshold_low': trp.threshold_low,
    '_trigger_delays': trp.trigger_delays,
    '_triggered_channels': trp.triggered_channels,
    '_primary_trigger': trp.primary,
    '_sample_block_size': trp.sample_block_size,
    '_gaussian_jitter': trp.gaussian_jitter,
    '_threshold_factor': trp.threshold_factor,
    '_number_of_coincidences': trp.number_of_coincidences,
    '_high_low_window': trp.high_low_window,
    '_integration_window': trp.integration_window,
    '_power_mean': trp.power_mean,
    '_power_std': trp.power_std,
    '_passband': trp.passband,
    '_output_passband': trp.output_passband,
    '_order': trp.filter_order,
    '_phasing_angles': trp.phasing_angles,
    '_primary_angles': trp.phasing_angles,
    '_maximum_amps': trp.maximum_amps,
    '_window_size': trp.window_size,
    '_step_size': trp.step_size,
    '_step_side': trp.step_size,  # typo in files written before NuRadioReco 3.2
    '_temperature': trp.temperature,
    '_Vbias': trp.bias_voltage,
    '_secondary_channels': trp.secondary_channels,
    '_secondary_angles': trp.secondary_angles,
    '_sec_trigger_delays': trp.sec_trigger_delays,
}

# Legacy attributes that were copies of `_channels` (dropped)
_LEGACY_DUPLICATES = ('_primary_channels', '_trigger_channels')

_warned_legacy_attributes = set()


class Trigger(ParameterStorage):
    """
    base class to store different triggers

    Subclasses register their trigger type string with ``class MyTrigger(Trigger, trigger_type='my_type')``.
    """

    _TRIGGER_TYPE = 'default'

    def __init_subclass__(cls, trigger_type=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if trigger_type is not None:
            cls._TRIGGER_TYPE = trigger_type
            _TRIGGER_CLASSES[trigger_type] = cls

    def __init__(self, name, channels=None, trigger_type='default', pre_trigger_times=55 * units.ns,
                 sample_block_size=0, gaussian_jitter=0 * units.ns):
        """
        initialize trigger class

        Parameters
        ----------
        name: string
            unique name of the trigger
        channels: array of ints
            the channels that are involved in the trigger
        trigger_type: string
            the trigger type
        pre_trigger_times: float or dict of floats
            the time before the trigger time that should be read out
            if a dict is given, the keys are the channel_ids, and the value is the pre_trigger_time between the
            start of the trace and the trigger time.
            if only a float is given, the same pre_trigger_time is used for all channels
        sample_block_size: int
            range of the uniform integer jitter (in samples) caused by the
            readout window being collected in sample blocks.
            An integer is drawn from ``[-sample_block_size / 2, sample_block_size / 2)``.
            Default: 0 samples(no block jitter)
        gaussian_jitter: positive int or float
            standard deviation of a Gaussian time smear applied to the
            readout window position (in samples, note check the sampling rate of the detector).
            Default: 0 samples (no Gaussian jitter)

        """
        ParameterStorage.__init__(self, trp)
        self._name = name
        self._trigger_type = trigger_type
        self[trp.channels] = channels
        self[trp.triggered] = False
        self[trp.trigger_time] = None
        self[trp.trigger_times] = None
        self[trp.triggered_channels] = []
        self.set_pre_trigger_times(pre_trigger_times)
        self[trp.primary] = False
        self.set_jitter_params(sample_block_size, gaussian_jitter)

    def set_primary(self, primary_trigger=True):
        """
        set the trigger to be the primary trigger.
        The trigger time of the primary trigger is used to define the readout window. There should only be one primary trigger.
        """
        self[trp.primary] = primary_trigger

    def is_primary(self):
        """
        return True if this trigger is the primary trigger
        """
        return self[trp.primary]

    def has_triggered(self):
        """
        returns true if station has trigger, false otherwise
        """
        return self[trp.triggered]

    def set_triggered(self, triggered=True):
        """ set the trigger to True or False """
        self[trp.triggered] = triggered

    def set_trigger_time(self, time):
        """ set the trigger time

        Parameters
        ----------
        time: float
            the trigger time from the beginning of the trace
        """
        self[trp.trigger_time] = time

    def get_trigger_time(self):
        """
        Get the trigger time.

        Returns trigger time, i.e. the first time in the event where the trigger condition was fulfilled.
        This is defined relative to the `station_time <NuRadioReco.framework.station.Station.get_station_time>`.

        Returns
        -------
        trigger_time : float
            The trigger time

        See Also
        --------
        get_trigger_times : function to return all times where the trigger condition was fulfilled
        """
        return self[trp.trigger_time]

    def set_trigger_times(self, times):
        """ set the trigger times

        Parameters
        ----------
        times: array
            all trigger times
        """
        self[trp.trigger_times] = times

    def get_trigger_times(self):
        """
        Get the trigger times

        For some triggers, in addition to the time of the first trigger,
        also all subsequent times where the trigger condition were fulfilled are
        stored. For triggers that do not store this information, this method
        is equivalent to `get_trigger_time` with the exception that it returns
        an array (of shape (1,)) instead of a scalar.
        Note that the trigger times are defined relative to the
        `station_time <NuRadioReco.framework.station.Station.get_station_time>`.


        Returns
        -------
        trigger_times : np.ndarray or None
            Array of all times where the trigger condition was satisfied.
            None if no trigger time is set.

        See Also
        --------
        get_trigger_time : method to return the (first) trigger time
        """
        trigger_times = self.get_parameter(trp.trigger_times, copy=False)
        trigger_time = self.get_trigger_time()
        if trigger_times is None and trigger_time is not None and not np.isnan(trigger_time):
            return np.array(trigger_time)
        return trigger_times

    def get_name(self):
        """ get trigger name """
        return self._name

    def get_type(self):
        """ get trigger type """
        return self._trigger_type

    def get_triggered_channels(self):
        """ get IDs of channels that have triggered """
        return self.get_parameter(trp.triggered_channels, copy=False)

    def set_triggered_channels(self, triggered_channels):
        self[trp.triggered_channels] = triggered_channels

    def set_pre_trigger_times(self, pre_trigger_times):
        """
        Set the pre-trigger times

        Parameters
        ----------
        pre_trigger_times: float or dict
            the time before the trigger time that should be read out
            if a dict is given, the keys are the channel_ids, and the value is the pre_trigger_time between the
            start of the trace and the trigger time.
            if only a float is given, the same pre_trigger_time is used for all channels
        """
        self[trp.pre_trigger_times] = pre_trigger_times

    def get_pre_trigger_times(self):
        """
        Return the pre_trigger_time between the start of the trace and the (global) trigger time

        Returns
        -------
        pre_trigger_times: dict | float
            the time before the trigger time that should be read out
            if a dict is returned, the keys are the channel_ids, and the value is the pre_trigger_time between the
            start of the trace and the trigger time.
            if only a float is given, the same pre_trigger_time is used for all channels

        """
        return self.get_parameter(trp.pre_trigger_times, copy=False)

    def get_pre_trigger_time_channel(self, channel_id):
        """
        get the trigger time for a specific channel

        convenience function to get the trigger time for a specific channel

        Parameters
        ----------
        channel_id: int
            the channel id

        Returns
        -------
        trigger_time: float
            the trigger time for the channel
        """
        pre_trigger_times = self.get_pre_trigger_times()
        if isinstance(pre_trigger_times, dict):
            return pre_trigger_times[channel_id]
        return pre_trigger_times

    def set_jitter_params(self, sample_block_size=0, gaussian_jitter=0):
        """
        Set the readout-window jitter parameters.

        Parameters
        ----------
        sample_block_size : int
            Range of the uniform integer jitter (in samples) caused by the
            readout window being collected in sample blocks.
            Default: 0 (no block jitter)
        gaussian_jitter : float
            Standard deviation of a Gaussian time smear applied to the
            readout window position (in samples).
            Default: 0 (no Gaussian jitter)
        """
        self[trp.sample_block_size] = sample_block_size
        self[trp.gaussian_jitter] = gaussian_jitter

    def get_sample_block_size(self):
        """Return the sample block size for readout-window jitter."""
        return self[trp.sample_block_size]

    def get_gaussian_jitter(self):
        """Return the Gaussian jitter spread for readout-window jitter."""
        return self[trp.gaussian_jitter]

    def serialize(self):
        data = ParameterStorage.serialize(self)
        data['_name'] = self._name
        data['_trigger_type'] = self._trigger_type
        return _dumps(data, protocol=4)

    def deserialize(self, data_pkl):
        self._set_state(pickle.loads(data_pkl))

    def _set_state(self, data):
        """ Set name, type and parameters from the unpickled output of `serialize` (or a legacy ``__dict__``). """
        self._name = data['_name']
        self._trigger_type = data['_trigger_type']
        if '_parameters' in data:
            self._parameters = {}
            self._parameter_covariances = {}
            ParameterStorage.deserialize(self, data)
        else:
            self._set_legacy_state(data)

    def _set_legacy_state(self, data):
        """
        Map the attributes of a trigger written before the triggerParameters were introduced
        (pickled ``__dict__``) onto the parameters. Values are taken as stored (no type conversion).
        Attributes without a parameter are dropped with a warning (once per trigger type and attribute).
        """
        for key, value in data.items():
            if key in ('_name', '_trigger_type') or key in _LEGACY_DUPLICATES:
                continue
            elif key in _LEGACY_ATTRIBUTES:
                self[_LEGACY_ATTRIBUTES[key]] = value
            elif key == '_coinc_window' and (value is None or isinstance(value, (int, float, np.number))):
                self[trp.coinc_window] = value
            else:
                # e.g. `_coinc_window` holding the channel list (bug in powerIntegration before NuRadioReco 3.2)
                if (self._trigger_type, key) not in _warned_legacy_attributes:
                    _warned_legacy_attributes.add((self._trigger_type, key))
                    logger.warning(
                        f"Trigger of type '{self._trigger_type}' from an old file has the attribute "
                        f"'{key}' ({value!r}) without a corresponding triggerParameter. It is dropped.")

    def __str__(self):
        output = ""
        for key, value in self.get_trigger_settings().items():
            output += "{}: {}\n".format(key, value)
        return output

    def show(self, print_stdout=True, **kwargs):
        """
        Print an overview of the structure of the Trigger.

        Other Parameters
        ----------------
        print_stdout : bool, optional
            If `True` (default), print the Trigger structure to stdout.
            Otherwise, return the string representation

        Returns
        -------
        str_output : str, optional
            A string representation of the Trigger structure.

        """
        self_string = (
            f"Trigger({self.get_name()}) ({self.get_type()}, "
            + ('primary, ' if self.is_primary() else '')
            + ('triggered, ' if self.has_triggered() else '')
            + (f'channels {self.get_triggered_channels()}')
            + ')')

        if print_stdout:
            print(self_string)
            return

        return self_string

    def get_trigger_settings(self):
        """
        Return name, type and all parameters of the trigger.

        Returns
        -------
        settings : dict
            Keys ``name``, ``trigger_type`` and the names of the set `triggerParameters`
            (e.g. ``triggered``, ``trigger_time``, ``triggered_channels``, ``threshold``). Values are not copied.
        """
        output = {'name': self._name, 'trigger_type': self._trigger_type}
        for key, value in self._parameters.items():
            output[key.name] = value
        return output


_TRIGGER_CLASSES['default'] = Trigger


class SimpleThresholdTrigger(Trigger, trigger_type='simple_threshold'):

    def __init__(self, name, threshold, channels=None, number_of_coincidences=1,
                 channel_coincidence_window=None, pre_trigger_times=55 * units.ns):
        """
        initialize trigger class

        Parameters
        ----------
        name: string
            unique name of the trigger
        threshold: float or dict of floats
            the threshold
        channels: array of ints or None
            the channels that are involved in the trigger
            default: None, i.e. all channels
        number_of_coincidences: int
            the number of channels that need to fulfill the trigger condition
            default: 1
        channel_coincidence_window: float or None (default)
            the coincidence time between triggers of different channels
        pre_trigger_times: float or dict of floats
            the time before the trigger time that should be read out
            if a dict is given, the keys are the channel_ids, and the value is the pre_trigger_time between the
            start of the trace and the trigger time.
            if only a float is given, the same pre_trigger_time is used for all channels

        """
        Trigger.__init__(self, name, channels, self._TRIGGER_TYPE, pre_trigger_times=pre_trigger_times)
        self[trp.threshold] = threshold
        self[trp.number_of_coincidences] = number_of_coincidences
        self[trp.coinc_window] = channel_coincidence_window


class EnvelopePhasedTrigger(Trigger, trigger_type='analog_envelope_phased'):

    def __init__(self, name, threshold_factor, power_mean, power_std,
                 triggered_channels=None, phasing_angles=None, trigger_delays=None,
                 output_passband=(None, None), pre_trigger_times=55 * units.ns):
        """
        initialize trigger class

        Parameters
        ----------
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
        pre_trigger_times: float or dict of floats
            the time before the trigger time that should be read out
            if a dict is given, the keys are the channel_ids, and the value is the pre_trigger_time between the
            start of the trace and the trigger time.
            if only a float is given, the same pre_trigger_time is used for all channels

        """
        Trigger.__init__(self, name, triggered_channels, self._TRIGGER_TYPE, pre_trigger_times=pre_trigger_times)
        self[trp.triggered_channels] = triggered_channels  # as before: the configured channels
        self[trp.phasing_angles] = phasing_angles
        self[trp.threshold_factor] = threshold_factor
        self[trp.power_mean] = power_mean
        self[trp.power_std] = power_std
        self[trp.trigger_delays] = trigger_delays
        self[trp.output_passband] = output_passband


_TRIGGER_CLASSES['envelope_phased'] = EnvelopePhasedTrigger  # type string used by older versions


class DigitalEnvelopePhasedTrigger(Trigger, trigger_type='digital_envelope_phased'):

    def __init__(self, name, threshold, trigger_channels=None,
                  phasing_angles=None, trigger_delays=None,
                  maximum_amps=None, pre_trigger_times=55 * units.ns):
        """
        initialize trigger class

        Parameters
        ----------
        name: string
            unique name of the trigger
        threshold: float
            the threshold
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
        Trigger.__init__(self, name, trigger_channels, self._TRIGGER_TYPE, pre_trigger_times=pre_trigger_times)
        self[trp.phasing_angles] = phasing_angles
        self[trp.threshold] = threshold
        self[trp.trigger_delays] = trigger_delays
        self[trp.maximum_amps] = maximum_amps


class SimplePhasedTrigger(Trigger, trigger_type='simple_phased'):

    def __init__(self, name, threshold, channels=None, secondary_channels=None,
                 primary_angles=None, secondary_angles=None,
                 trigger_delays=None, sec_trigger_delays=None,
                 window_size=None, step_size=None,
                 maximum_amps=None, pre_trigger_times=55 * units.ns
                ):
        """
        initialize trigger class

        Parameters
        ----------
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
        window_size: int
            the size of the integration window (units of ADC time ticks)
        step_size: int
            the size of the stride between calculating the phasing (units of ADC time ticks)
        maximum_amps: list of floats (length equal to that of `phasing_angles`)
            the maximum value of all the integration windows for each of the phased waveforms
        pre_trigger_times: float or dict of floats
            the time before the trigger time that should be read out
            if a dict is given, the keys are the channel_ids, and the value is the pre_trigger_time between the
            start of the trace and the trigger time.
            if only a float is given, the same pre_trigger_time is used for all channels

        """
        Trigger.__init__(self, name, channels, self._TRIGGER_TYPE, pre_trigger_times=pre_trigger_times)
        self[trp.phasing_angles] = primary_angles
        self[trp.secondary_channels] = secondary_channels
        self[trp.secondary_angles] = secondary_angles
        self[trp.threshold] = threshold
        self[trp.trigger_delays] = trigger_delays
        self[trp.sec_trigger_delays] = sec_trigger_delays
        self[trp.window_size] = window_size
        self[trp.step_size] = step_size
        self[trp.maximum_amps] = maximum_amps


class HighLowTrigger(Trigger, trigger_type='high_low'):

    def __init__(self, name, threshold_high, threshold_low, high_low_window,
                 channel_coincidence_window, channels=None, number_of_coincidences=1, pre_trigger_times=55 * units.ns):
        """
        initialize trigger class

        Parameters
        ----------
        name: string
            unique name of the trigger
        threshold_high: float or dict of floats
            the high threshold
        threshold_low: float or dict of floats
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
        pre_trigger_times: float or dict of floats
            the time before the trigger time that should be read out
            if a dict is given, the keys are the channel_ids, and the value is the pre_trigger_time between the
            start of the trace and the trigger time.
            if only a float is given, the same pre_trigger_time is used for all channels

        """
        Trigger.__init__(self, name, channels, self._TRIGGER_TYPE, pre_trigger_times=pre_trigger_times)
        self[trp.number_of_coincidences] = number_of_coincidences
        self[trp.threshold_high] = threshold_high
        self[trp.threshold_low] = threshold_low
        self[trp.high_low_window] = high_low_window
        self[trp.coinc_window] = channel_coincidence_window


class IntegratedPowerTrigger(Trigger, trigger_type='int_power'):

    def __init__(self, name, threshold, channel_coincidence_window, channels=None, number_of_coincidences=1,
                 power_mean=None, power_std=None, integration_window=None, pre_trigger_times=55 * units.ns):
        """
        initialize trigger class

        Parameters
        ----------
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
        pre_trigger_times: float or dict of floats
            the time before the trigger time that should be read out
            if a dict is given, the keys are the channel_ids, and the value is the pre_trigger_time between the
            start of the trace and the trigger time.
            if only a float is given, the same pre_trigger_time is used for all channels

        """
        Trigger.__init__(self, name, channels, self._TRIGGER_TYPE, pre_trigger_times=pre_trigger_times)
        self[trp.number_of_coincidences] = number_of_coincidences
        self[trp.threshold] = threshold
        self[trp.coinc_window] = channel_coincidence_window
        self[trp.power_mean] = power_mean
        self[trp.power_std] = power_std
        self[trp.integration_window] = integration_window


class EnvelopeTrigger(Trigger, trigger_type='envelope_trigger'):

    def __init__(self, name, passband, order, threshold, number_of_coincidences=2,
                 channel_coincidence_window=None, channels=None, pre_trigger_times=55 * units.ns):
        """
        initialize trigger class

        Parameters
        ----------
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
        pre_trigger_times: float or dict of floats
            the time before the trigger time that should be read out
            if a dict is given, the keys are the channel_ids, and the value is the pre_trigger_time between the
            start of the trace and the trigger time.
            if only a float is given, the same pre_trigger_time is used for all channels

        """
        Trigger.__init__(self, name, channels, self._TRIGGER_TYPE, pre_trigger_times=pre_trigger_times)
        self[trp.passband] = passband
        self[trp.filter_order] = order
        self[trp.threshold] = threshold
        self[trp.number_of_coincidences] = number_of_coincidences
        self[trp.coinc_window] = channel_coincidence_window


class RNOGSurfaceTrigger(Trigger, trigger_type='rnog_surface_trigger'):

    def __init__(self, name, threshold, number_of_coincidences=1,
                 channel_coincidence_window=60*units.ns, channels=[13, 16, 19],
                 temperature=250*units.kelvin, Vbias=2*units.volt, pre_trigger_times=55 * units.ns):
        """
        initialize trigger class

        Parameters
        ----------
        name: string
            unique name of the trigger
        threshold: float
            the threshold
        number_of_coincidences: int
            the number of channels that need to fulfill the trigger condition
            default: 1
        channel_coincidence_window: float or None (default)
            the coincidence time between triggers of different channels
        channels: array of ints or None
            the channels that are involved in the trigger
            default: None, i.e. all channels
        temperature: float
            temperature of the trigger board
        Vbias: float
            bias voltage on the trigger board
        pre_trigger_times: float or dict of floats
            the time before the trigger time that should be read out
            if a dict is given, the keys are the channel_ids, and the value is the pre_trigger_time between the
            start of the trace and the trigger time.
            if only a float is given, the same pre_trigger_time is used for all channels

        """
        Trigger.__init__(self, name, channels, self._TRIGGER_TYPE, pre_trigger_times=pre_trigger_times)
        self[trp.threshold] = threshold
        self[trp.number_of_coincidences] = number_of_coincidences
        self[trp.coinc_window] = channel_coincidence_window
        self[trp.temperature] = temperature
        self[trp.bias_voltage] = Vbias
