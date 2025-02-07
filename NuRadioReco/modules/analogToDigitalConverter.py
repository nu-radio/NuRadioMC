import logging
import time
import numpy as np
from NuRadioReco.utilities import units
from scipy.interpolate import interp1d
import scipy.signal
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities.trace_utilities import delay_trace


def perfect_comparator(trace, adc_n_bits, adc_voltage_range, mode='floor', output='voltage'):
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
    adc_voltage_range: float
        Voltage range V_range = V_max - V_min where V_max corresponds to the maximum number of counts given by the
        ADC 2**adc_n_bits - 1 and V_min to the ADC count 0.
    mode: string
        'floor' or 'ceiling'
    output: {'voltage', 'counts'}, default 'voltage'
        Options:

        * 'voltage' to store the ADC output as discretised voltage trace
        * 'counts' to store the ADC output in ADC counts

    Returns
    -------
    digital_trace: array of floats
        Digitised voltage trace in volts or ADC counts
    """

    lsb_voltage = adc_voltage_range / (2 ** adc_n_bits- 1)

    if mode == 'floor':
        digital_trace = np.floor(trace / lsb_voltage).astype(int)
    elif mode == 'ceiling':
        digital_trace = np.ceil(trace / lsb_voltage).astype(int)
    else:
        raise ValueError('Choose floor or ceiing as modes for the comparator ADC')

    digital_trace = apply_saturation(digital_trace, adc_n_bits)

    if output == 'voltage':
        digital_trace = lsb_voltage * digital_trace.astype(float)
    elif output == 'counts':
        pass
    else:
        raise ValueError("The ADC output format is unknown. Please choose 'voltage' or 'counts'")

    return digital_trace


def perfect_floor_comparator(trace, adc_n_bits, adc_voltage_range, output='voltage'):
    """
    Perfect comparator ADC that takes the floor value of the comparison.
    See perfect_comparator
    """
    return perfect_comparator(trace, adc_n_bits, adc_voltage_range, mode='floor', output=output)


def perfect_ceiling_comparator(trace, adc_n_bits, adc_voltage_range, output='voltage'):
    """
    Perfect comparator ADC that takes the floor value of the comparison.
    See perfect_floor.
    """
    return perfect_comparator(trace, adc_n_bits, adc_voltage_range, mode='ceiling', output=output)


def apply_saturation(adc_counts_trace, adc_n_bits):
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
    Returns
    -------
    saturated_trace: array of floats
        The clipped or saturated voltage trace
    """
    # This function assumes that the digitized trace has counts in the range from -2**(adc_n_bits-1) to 2**(adc_n_bits-1)-1.
    # Hence, if the trace has no negative entries, it is very likely that the trace is wrong (i.e., was maybe converted with a wrong convention).
    if not np.any(adc_counts_trace < 0):
        raise ValueError("The ADC trace has no negative entries. This is very likely wrong!")

    highest_count = 2 ** (adc_n_bits - 1) - 1
    high_saturation_mask = adc_counts_trace > highest_count
    adc_counts_trace[high_saturation_mask] = highest_count

    lowest_count = -2 ** (adc_n_bits - 1)
    low_saturation_mask = adc_counts_trace < lowest_count
    adc_counts_trace[low_saturation_mask] = lowest_count

    return adc_counts_trace


class analogToDigitalConverter:
    """
    This class simulates an analog to digital converter. The steps followed
    by this module to achieve the conversion are:

    1) The following properties of the channel are read. They must be in the
        detector configuration file:

        * "adc_nbits", the number of bits of the ADC
        * "adc_reference_voltage", the reference voltage in volts, that is, the
            maximum voltage the ADC can convert without saturating
        * "adc_sampling_frequency", the sampling frequency in GHz

    2) A random clock offset (jitter) can be added, as it would happen in
        a real experiment. Choose random_clock_offset = True to do so. A time
        delay can also be fixed if the field "adc_time_delay" is specified in ns.
        The channel trace is interpolated to get the trace values at the clock
        times displaced from the channel times. This is fine as long as the input
        channel traces have been simulated with a sampling rate greater than the
        ADC sampling rate, which should be the case. Upsampling is also possible,
        and recommended for phased array simulations.

        .. Important:: Upsampling after digitisation is performed by the FPGA, which
            means that the digitised trace is no longer discretised after being upsampled.
            The FPGA uses fixed point arithmetic, which in practice can be approximated
            as floats for our simulation purposes.

    3) A type of ADC converter is chosen, which transforms the trace in ADC
        counts (discrete values). The available types are listed in the list
        _adc_types, which are (see functions with the same names for documentation):

        * 'perfect_floor_comparator'
        * 'perfect_ceiling_comparator'

    .. Important:: Since this module already performs a downsampling, there is no
        need to use the channelResampler in those channels that possess an ADC. The
        chosen method for resampling is interpolation, since filtering only the
        spectrum below half the sampling frequency would eliminate the higher Nyquist
        zones.

    Note that after this module the traces are still expressed in voltage units,
    only the possible values are discretised.

    If the ADC is used for triggering and the user does not want to modify the
    trace, the function get_digital_trace can be used. If there are two different
    ADCs for the same channel, one for triggering and another one for storing,
    one can define a trigger ADC adding `"trigger_"` to every relevant ADC field
    in the detector configuration, and use them setting `trigger_adc` to True in
    `get_digital_trace`.

    """

    def __init__(self):
        self.__t = 0
        self._adc_types = {
            'perfect_floor_comparator': perfect_floor_comparator,
            'perfect_ceiling_comparator': perfect_ceiling_comparator}

        self._mandatory_fields = ['adc_nbits', 'adc_sampling_frequency']

        self.logger = logging.getLogger('NuRadioReco.analogToDigitalConverter')

    def _get_adc_parameters(self, det_channel, vrms=None, trigger_adc=False):
        """ Get the ADC parameters for a channel from the detector description """

        field_prefix = 'trigger_' if trigger_adc else ''
        for field in self._mandatory_fields:
            field_check = field_prefix + field
            if field_check not in det_channel:
                raise ValueError(
                    f"The field {field_check} is not present in channel {channel_id}. "
                    "Please specify it on your detector file.")

        # Use "or" if "field_prefix + "adc_time_delay"" is in dict but value is None
        adc_time_delay = (det_channel.get(field_prefix + "adc_time_delay", 0) or 0) * units.ns

        adc_n_bits = det_channel[field_prefix + "adc_nbits"]
        adc_sampling_frequency = det_channel[field_prefix + "adc_sampling_frequency"] * units.GHz

        if vrms is None:
            if field_prefix + "adc_reference_voltage" in det_channel:
                error_msg = (
                    f"The field \"{field_prefix}adc_reference_voltage\" is present in channel {channel_id}. "
                    "This field is deprecated. Please use the field adc_voltage_range instead. However, "
                    "be aware that the definition of the two fields are different. The \"adc_reference_voltage\" "
                    "referred to the maximum voltage V_max of the maximum ADC count 2^n - 1, assuming that the "
                    "ADC operates from -V_max to V_max. The \"adc_voltage_range\" refers to the maximum voltage "
                    "range V_range = V_max - V_min where V_max corresponds to 2^n - 1 and V_min to 0. Example: "
                    "If you want to simulate a ADC with a dynamic range from -1 V to 1 V, you should set have set "
                    "the adc_reference_voltage to 1 V. Now you have to set the adc_voltage_range to 2 V."
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg)


            adc_ref_voltage_label = field_prefix + "adc_voltage_range"
            if adc_ref_voltage_label not in det_channel:
                raise ValueError(
                    f"The field {adc_ref_voltage_label} is not present in channel {channel_id}. "
                    "Please specify it on your detector file.")

            adc_ref_voltage = det_channel[adc_ref_voltage_label] * units.V
        else:
            adc_noise_nbits_label = field_prefix + "adc_noise_nbits"
            if adc_noise_nbits_label not in det_channel:
                raise ValueError(
                    f"The field {adc_noise_nbits_label} is not present in channel {channel_id}. "
                    "Please specify it on your detector file.")

            adc_noise_n_bits = det_channel[adc_noise_nbits_label]
            adc_ref_voltage = vrms * (2 ** adc_n_bits - 1) / (2 ** (adc_noise_n_bits - 1))

        return adc_n_bits, adc_ref_voltage, adc_sampling_frequency, adc_time_delay

    def get_digital_trace(self, station, det, channel,
                          Vrms=None,
                          trigger_adc=False,
                          clock_offset=0.0,
                          adc_type='perfect_floor_comparator',
                          return_sampling_frequency=False,
                          adc_output='voltage',
                          trigger_filter=None):
        """
        Returns the digital trace for a channel, without setting it. This allows
        the creation of a digital trace that can be used for triggering purposes
        without removing the original information on the channel.

        Parameters
        ----------
        station: framework.station.Station object
        det: detector.detector.Detector object
        channel: framework.channel.Channel object
        Vrms: float
            If supplied, overrides adc_reference_voltage as supplied in the detector description file
        trigger_adc: bool
            If True, the relevant ADC parameters in the config file are the ones
            that start with `'trigger_'`
        random_clock_offset: bool
            If True, a random clock offset between -1 and 1 clock cycles is added
        adc_type: string
            The type of ADC used. The following are available:

            * perfect_floor_comparator
            * perfect_ceiling_comparator

            See functions with the same name on this module for documentation
        return_sampling_frequency: bool
            If True, returns the trace and the ADC sampling frequency
        adc_output: string
            Options:

            * 'voltage' to store the ADC output as discretised voltage trace
            * 'counts' to store the ADC output in ADC counts

        trigger_filter: array floats
            Freq. domain of the response to be applied to post-ADC traces
            Must be length for "MC freq"

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
        adc_n_bits, adc_ref_voltage, adc_sampling_frequency, adc_time_delay = self._get_adc_parameters(
            det_channel, vrms=Vrms, trigger_adc=trigger_adc)

        if clock_offset:
            adc_time_delay += clock_offset / adc_sampling_frequency

        sampling_rate = channel.get_sampling_rate()
        if adc_sampling_frequency > channel.get_sampling_rate():
            raise ValueError(
                'The ADC sampling rate is greater than '
                f'the channel {channel.get_id()} sampling rate. '
                'Please change the ADC sampling rate.')

        if trigger_filter is not None:
            apply_filter(channel, trigger_filter)

        if adc_time_delay:
            # Random clock offset
            trace, dt_tstart = delay_trace(channel, sampling_rate, adc_time_delay)
            times = channel.get_times()
            if dt_tstart > 0:
                # by design dt_tstart is a multiple of the sampling rate
                times = times[int(round(dt_tstart / sampling_rate)):]
            times = times[:len(trace)]
            channel.set_trace(trace, sampling_rate, trace_start_time=times[0])

        # Upsampling to 5 GHz before downsampling using interpolation.
        # We cannot downsample with a Fourier method because we want to keep
        # the higher Nyquist zones.
        upsampling_frequency = 5.0 * units.GHz
        if upsampling_frequency > sampling_rate:
            channel.resample(upsampling_frequency)

        # Downsampling to ADC frequency
        resampled_times, resampled_trace = downsampling_linear_interpolation(
            channel.get_trace(), sampling_rate, adc_sampling_frequency)
        resampled_times += channel.get_trace_start_time()

        # Digitisation
        digital_trace = self._adc_types[adc_type](resampled_trace, adc_n_bits, adc_ref_voltage, adc_output)

        # Ensuring trace has an even number of samples
        if len(digital_trace) % 2 == 1:
            digital_trace = digital_trace[:-1]

        if return_sampling_frequency:
            return digital_trace, adc_sampling_frequency
        else:
            return digital_trace

    @register_run()
    def run(self, evt, station, det,
            clock_offset=0.0,
            adc_type='perfect_floor_comparator',
            adc_output='voltage',
            trigger_filter=None):
        """
        Runs the analogToDigitalConverter and transforms the traces from all
        the channels of an input station to digital voltage values.

        Parameters
        ----------
        evt: framework.event.Event object
        station: framework.station.Station object
        det: detector.detector.Detector object
        clock_offset: float

        adc_type: string
            The type of ADC used. The following are available:

            * 'perfect_floor_comparator'

            See functions with the same name on this module for documentation
        adc_output: string
            Options:

            * 'voltage' to store the ADC output as discretised voltage trace
            * 'counts' to store the ADC output in ADC counts

        upsampling_factor: integer
            Upsampling factor. The digital trace will be a upsampled to a
            sampling frequency int_factor times higher than the original one

        """

        t = time.time()

        for channel in station.iter_channels():
            digital_trace, adc_sampling_frequency = self.get_digital_trace(
                station, det, channel,
                clock_offset=clock_offset,
                adc_type=adc_type,
                return_sampling_frequency=True,
                adc_output=adc_output,
                trigger_filter=trigger_filter
            )

            channel.set_trace(digital_trace, adc_sampling_frequency)

        self.__t += time.time() - t

    def end(self):
        from datetime import timedelta
        self.logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        self.logger.info("total time used by this module is {}".format(dt))
        return dt


def downsampling_linear_interpolation(trace, sampling_rate, new_sampling_rate):
    """
    Downsamples a trace using linear interpolation.

    Parameters
    ----------
    trace: array of floats
        The trace to be downsampled
    sampling_rate: float
        The sampling rate of the input trace
    new_sampling_rate: float
        The sampling rate of the output trace

    Returns
    -------
    times_downsampled: array of floats
        The times of the downsampled trace (without start time)
    downsampled_trace: array of floats
        The downsampled trace
    """

    if new_sampling_rate >= sampling_rate:
        raise ValueError('The new sampling rate must be lower than the original one')

    n_samples = int((new_sampling_rate / sampling_rate) * len(trace))
    times = np.arange(len(trace)) / sampling_rate
    times_downsampled = np.arange(n_samples) / new_sampling_rate

    # downsampled_trace = np.interp(
    #   resampled_times, times, trace, left=trace[0], right=trace[-1])
    interpolate_trace = interp1d(
        times, trace, kind='linear', fill_value=(trace[0], trace[-1]), bounds_error=False)
    downsampled_trace = interpolate_trace(times_downsampled)

    return times_downsampled, downsampled_trace


def apply_filter(channel, filter):
    """
    Applies a filter to a trace in the frequency domain.

    Parameters
    ----------
    channel: `NuRadioReco.framework.channel.Channel` object
        The (channel) trace to be filtered
    filter: array of floats
        The filter to be applied
    """
    channel.set_frequency_spectrum(
        channel.get_frequency_spectrum() * filter, "same"
    )
