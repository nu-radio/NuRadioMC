import logging
import time
import numpy as np
import NuRadioReco.modules.channelResampler
from NuRadioReco.utilities import units
from scipy.interpolate import interp1d
from scipy.signal import resample
from NuRadioReco.modules.base.module import register_run

def perfect_comparator(trace, adc_n_bits, adc_ref_voltage, mode='floor', output='voltage'):
    """
    Simulates a perfect comparator flash ADC that compares the voltage to the
    voltage for the least significative bit and takes the floor or the ceiling
    of their ratio as a digitised value of the trace.

    Parameters
    ----------
    trace: array of floats
        Trace containing the voltage to be digitised
    adc_n_bits: int
        Number of bits of the ADC
    adc_ref_voltage: float
        Voltage corresponding to the maximum number of counts given by the
        ADC: 2**adc_n_bits - 1
    mode: string
        'floor' or 'ceiling'
    output: string
        - 'voltage' to store the ADC output as discretised voltage trace
        - 'counts' to store the ADC output in ADC counts

    Returns
    -------
    digital_trace: array of floats
        Digitised voltage trace in volts or ADC counts
    """

    lsb_voltage = adc_ref_voltage/(2 ** (adc_n_bits-1) - 1)

    if (mode == 'floor'):
        digital_trace = np.floor(trace / lsb_voltage)
    elif (mode == 'ceiling'):
        digital_trace = np.ceil(trace / lsb_voltage)
    else:
        raise ValueError('Choose floor or ceiing as modes for the comparator ADC')

    digital_trace = apply_saturation(digital_trace, adc_n_bits, adc_ref_voltage)
    digital_trace = round_to_int(digital_trace)

    if (output == 'voltage'):
        digital_trace = lsb_voltage * digital_trace.astype(np.float)
    elif (output == 'counts'):
        pass
    else:
        raise ValueError("The ADC output format is unknown. Please choose 'voltage' or 'counts'" )

    return digital_trace

def perfect_floor_comparator(trace, adc_n_bits, adc_ref_voltage, output='voltage'):
    """
    Perfect comparator ADC that takes the floor value of the comparison.
    See perfect_comparator
    """

    return perfect_comparator(trace, adc_n_bits, adc_ref_voltage, mode='floor', output=output)

def perfect_ceiling_comparator(trace, adc_n_bits, adc_ref_voltage, output='voltage'):
    """
    Perfect comparator ADC that takes the floor value of the comparison.
    See perfect_floor.
    """

    return perfect_comparator(trace, adc_n_bits, adc_ref_voltage, mode='ceiling', output=output)

def apply_saturation(adc_counts_trace, adc_n_bits, adc_ref_voltage):
    """
    Takes a digitised trace in ADC counts and clips the parts of the
    trace with values higher than 2**(adc_n_bits-1)-1 or lower than
    -2**(adc_n_bits-1).

    Parameters
    ----------
    adc_counts_trace: array of floats
        Voltage in ADC counts, unclipped
    adc_n_bits: int
        Number of bits of the ADC
    adc_ref_voltage: float
        Voltage corresponding to the maximum number of counts given by the
        ADC: 2**(adc_n_bits-1) - 1

    Returns
    -------
    saturated_trace: array of floats
        The clipped or saturated voltage trace
    """

    saturated_trace = adc_counts_trace[:]

    highest_count = 2 ** (adc_n_bits-1) - 1
    high_saturation_mask = adc_counts_trace > highest_count
    saturated_trace[high_saturation_mask] = highest_count

    lowest_count = - 2 ** (adc_n_bits-1)
    low_saturation_mask = adc_counts_trace < lowest_count
    saturated_trace[low_saturation_mask] = lowest_count

    return saturated_trace

def round_to_int(digital_trace):

    int_trace = np.rint(digital_trace)
    int_trace = int_trace.astype(int)

    return int_trace

class analogToDigitalConverter():
    """
    This class simulates an analog to digital converter. The steps followed
    by this module to achieve the conversion are:
    1) The following properties of the channel are read. They must be in the
    detector configuration file:
        - "adc_nbits", the number of bits of the ADC
        - "adc_reference_voltage", the reference voltage in volts, that is, the
        maximum voltage the ADC can convert without saturating
        - "adc_sampling_frequency", the sampling frequency in GHz
    2) A random clock offset (jitter) can be added, as it would happen in
    a real experiment. Choose random_clock_offset = True to do so. A time
    delay can also be fixed if the field "adc_time_delay" is specified in ns.
    The channel trace is interpolated to get the trace values at the clock
    times displaced from the channel times. This is fine as long as the input
    channel traces have been simulated with a sampling rate greater than the
    ADC sampling rate, which should be the case.
    3) A type of ADC converter is chosen, which transforms the trace in ADC
    counts (discrete values). The available types are listed in the list
    _adc_types, which are (see functions with the same names for documentation):
        - 'perfect_floor_comparator'
        - 'perfect_ceiling_comparator'

    IMPORTANT: Since this module already performs a downsampling, there is no
    need to use the channelResampler in those channels that possess an ADC.

    Note that after this module the traces are still expressed in voltage units,
    only the possible values are discretised.

    If the ADC is used for triggering and the user does not want to modify the
    trace, the function get_digital_trace can be used. If there are two different
    ADCs for the same channel, one for triggering and another one for storing,
    one can define a trigger ADC adding "trigger_" to every relevant ADC field
    in the detector configuration, and use them setting trigger_adc to True in
    get_digital_trace.
    """

    def __init__(self):
        self.__t = 0
        self._adc_types = {'perfect_floor_comparator': perfect_floor_comparator,
                           'perfect_ceiling_comparator': perfect_ceiling_comparator}
        self._mandatory_fields = ['adc_nbits',
                                  'adc_reference_voltage',
                                  'adc_sampling_frequency']

        self.logger = logging.getLogger('NuRadioReco.analogToDigitalConverter')

    def get_digital_trace(self, station, det, channel,
                          trigger_adc=False,
                          random_clock_offset=True,
                          adc_type='perfect_floor_comparator',
                          diode=None,
                          return_sampling_frequency=False,
                          output='voltage'):
        """
        Returns the digital trace for a channel, without setting it. This allows
        the creation of a digital trace that can be used for triggering purposes
        without removing the original information on the channel.

        Parameters
        ----------
        station: framework.station.Station object
        det: detector.detector.Detector object
        channel: framework.channel.Channel object
        trigger_adc: bool
            If True, the relevant ADC parameters in the config file are the ones
            that start with 'trigger_'
        random_clock_offset: bool
            If True, a random clock offset between -1 and 1 clock cycles is added
        adc_type: string
            The type of ADC used. The following are available:
            - perfect_floor_comparator
            - perfect_ceiling_comparator
            See functions with the same name on this module for documentation
        diode: utilities.diodeSimulator.diodeSimulator object
            Diode used to envelope filter the signal
        return_sampling_frequency: bool
            If True, returns the trace and the ADC sampling frequency
        output: string
            - 'voltage' to store the ADC output as discretised voltage trace
            - 'counts' to store the ADC output in ADC counts

        Returns
        -------
        digital_trace: array of floats
            Digitised voltage trace
        adc_sampling_frequency: float
            ADC sampling frequency for the channel
        """

        station_id = station.get_id()
        channel_id = channel.get_id()
        det_channel = det.get_channel(station_id, channel_id)

        for field in self._mandatory_fields:
            if trigger_adc:
                field_check = 'trigger_' + field
            else:
                field_check = field
            if field_check not in det_channel:
                channel_id = channel.get_id()
                error_msg  = "The field {} is not present in channel {}. ".format(field_check,
                                                                                  channel_id)
                error_msg += "Please specify it on your detector file"
                raise ValueError(error_msg)

        times = channel.get_times()[:]
        trace = channel.get_trace()[:]
        if diode is not None:
            trace = diode.tunnel_diode(channel)

        if trigger_adc:
            adc_time_delay_label = "trigger_adc_time_delay"
            adc_n_bits_label = "trigger_adc_nbits"
            adc_ref_voltage_label = "trigger_adc_reference_voltage"
            adc_sampling_frequency_label = "trigger_adc_sampling_frequency"
        else:
            adc_time_delay_label = "adc_time_delay"
            adc_n_bits_label = "adc_nbits"
            adc_ref_voltage_label = "adc_reference_voltage"
            adc_sampling_frequency_label = "adc_sampling_frequency"

        adc_time_delay = 0

        if adc_time_delay_label in det_channel:
            if det_channel[adc_time_delay_label] is not None:
                adc_time_delay = channel[adc_time_delay_label] * units.ns

        if random_clock_offset:
            clock_offset = np.random.uniform(-1, 1)
            adc_time_delay += clock_offset / channel.get_sampling_rate()

        adc_n_bits = det_channel[adc_n_bits_label]
        adc_ref_voltage = det_channel[adc_ref_voltage_label] * units.V
        adc_sampling_frequency = det_channel[adc_sampling_frequency_label] * units.GHz

        if (adc_sampling_frequency > channel.get_sampling_rate()):
            error_msg  = 'The ADC sampling rate is greater than '
            error_msg += 'the channel {} sampling rate. '.format(channel.get_id())
            error_msg += 'Please change the ADC sampling rate.'
            raise ValueError(error_msg)

        delayed_times = times + adc_time_delay
        interpolate_trace = interp1d(times, trace, kind='quadratic',
                                     fill_value='extrapolate')

        delayed_trace = interpolate_trace(delayed_times)

        new_n_samples = int( (adc_sampling_frequency / channel.get_sampling_rate()) * len(delayed_trace) )
        resampled_trace = resample(delayed_trace, new_n_samples)

        digital_trace = self._adc_types[adc_type](delayed_trace, adc_n_bits,
                                                  adc_ref_voltage, output)

        if return_sampling_frequency:
            return digital_trace, adc_sampling_frequency
        else:
            return digital_trace

    @register_run()
    def run(self, evt, station, det,
            random_clock_offset=True,
            adc_type='perfect_floor_comparator',
            output='voltage'):
        """
        Runs the analogToDigitalConverter and transforms the traces from all
        the channels of an input station to digital voltage values.

        Parameters
        ----------
        evt: framework.event.Event object
        station: framework.station.Station object
        det: detector.detector.Detector object
        random_clock_offset: bool
            If True, a random clock offset between -1 and 1 clock cycles is added
        adc_type: string
            The type of ADC used. The following are available:
            - perfect_floor_comparator
            See functions with the same name on this module for documentation
        output: string
            - 'voltage' to store the ADC output as discretised voltage trace
            - 'counts' to store the ADC output in ADC counts
        """

        t = time.time()

        for channel in station.iter_channels():

            digital_trace, adc_sampling_frequency = self.get_digital_trace(station, det, channel,
                                                        random_clock_offset=random_clock_offset,
                                                        adc_type=adc_type,
                                                        return_sampling_frequency=True,
                                                        output=output)

            channel.set_trace(digital_trace, adc_sampling_frequency)

        self.__t += time.time() - t


    def end(self):
        from datetime import timedelta
        self.logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        logger.info("total time used by this module is {}".format(dt))
        return dt
