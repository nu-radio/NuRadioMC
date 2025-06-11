from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units, signal_processing

import scipy.interpolate
import scipy.signal

import numpy as np
import time

import logging
logger = logging.getLogger('NuRadioReco.analogToDigitalConverter')


def perfect_comparator(trace, adc_n_bits, adc_voltage_range, output='voltage', mode_func=np.floor):
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
    adc_voltage_range: (float, float)
        Is a tuple (Vmin, Vmax) defining the "full scale" voltage range where V_max corresponds to the maximum number of counts given by the
        ADC 2**adc_n_bits - 1 and V_min to the ADC count 0.
    output: {'voltage', 'counts'}, default 'voltage'
        Options:

        * 'voltage' to store the ADC output as discretised voltage trace
        * 'counts' to store the ADC output in ADC counts

    mode_func: callable
        Either ``np.floor`` or ``np.ceil`` to choose the mode of the comparator

    Returns
    -------
    digital_trace: array of floats
        Digitised voltage trace in volts or ADC counts

    Notes
    -----
    There is often some ambiguity in the definition of the "Least Significant Bit" (i.e., resolution) for an ADC.
    People either define it as lsb = V / 2^n or lsb = V / (2^n - 1). As you can see we are adopting the latter. The ambiguity
    comes from an amibguity of what the symbol V in the prev. equations acutally is. The following link provides an explanation [1].
    In short: If you divide by 2^n, V refers to a reference voltage which can never be reached. If you divide by 2^n - 1 V refers to
    the "full scale" voltage which can be reached. How your ADC works will be defined in its datasheet.

    [1]: https://masteringelectronicsdesign.com/an-adc-and-dac-least-significant-bit-lsb/
    """

    lsb_voltage = (adc_voltage_range[1] - adc_voltage_range[0]) / (2 ** adc_n_bits - 1)
    logger.debug("LSB voltage: {:.2f} mV".format(lsb_voltage / units.mV))

    assert mode_func in [np.floor, np.ceil], "Choose floor or ceiing as modes for the comparator ADC"

    digital_trace = mode_func((trace - adc_voltage_range[0]) / lsb_voltage).astype(int)
    v_min_adc = mode_func(adc_voltage_range[0] / lsb_voltage).astype(int)

    digital_trace = apply_saturation(digital_trace, adc_n_bits)
    digital_trace += v_min_adc

    if output == 'voltage':
        digital_trace = lsb_voltage * digital_trace.astype(float)
    elif output == 'counts':
        pass
    else:
        raise ValueError("The ADC output format is unknown. Please choose 'voltage' or 'counts'")

    return digital_trace


def perfect_floor_comparator(*args, **kwargs):
    """
    Perfect comparator ADC that takes the floor value of the comparison.
    See perfect_comparator
    """
    return perfect_comparator(*args, **kwargs, mode_func=np.floor)


def perfect_ceiling_comparator(*args, **kwargs):
    """
    Perfect comparator ADC that takes the floor value of the comparison.
    See perfect_comparator.
    """
    return perfect_comparator(*args, **kwargs, mode_func=np.ceil)


def apply_saturation(adc_counts_trace, adc_n_bits):
    """
    Takes a digitised trace in ADC counts and clips the parts of the
    trace with values higher than 2**adc_n_bits - 1 or lower than 0.

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
    highest_count = 2 ** adc_n_bits - 1
    adc_counts_trace = np.where(adc_counts_trace > highest_count, highest_count, adc_counts_trace)
    adc_counts_trace = np.where(adc_counts_trace < 0, 0, adc_counts_trace)
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

    def __init__(self, log_level=logging.NOTSET):
        logger.setLevel(log_level)
        self._adc_types = {
            'perfect_floor_comparator': perfect_floor_comparator,
            'perfect_ceiling_comparator': perfect_ceiling_comparator}

        self._mandatory_fields = ['adc_nbits', 'adc_sampling_frequency']

    def _get_adc_parameters(self, det_channel, channel_id, vrms=None, trigger_adc=False):
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
                    f"This field is deprecated. Please use the field \"{field_prefix}adc_voltage_range\" instead. However, "
                    "be aware that the definition of the two fields are different. The \"adc_reference_voltage\" "
                    "referred to the maximum voltage V_max of the maximum ADC count 2^n - 1 with n being \"adc_nbits\", "
                    "assuming that the ADC operates from -V_max to V_max. As a consquence the voltage per ADC count "
                    "was calculated using: dV = adc_reference_voltage / (2^(n - 1) - 1). The \"adc_voltage_range\" "
                    "refers to the maximum voltage range V_range = V_max - V_min where V_max corresponds to 2^n - 1 "
                    "and V_min to 0. The voltage per ADC count was calculated using: dV = adc_voltage_range / (2^n - 1). "
                    "Example: If you want to simulate a ADC with a dynamic range from -1 V to 1 V, you should set have set "
                    "the adc_reference_voltage to 1 V. Now you have to set the adc_voltage_range to 2 V."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            if field_prefix + "adc_min_voltage" not in det_channel and field_prefix + "adc_max_voltage" not in det_channel:
                raise ValueError(
                    f"The fields \"{field_prefix}adc_min_voltage\" and \"{field_prefix}adc_max_voltage\" "
                    f"are not present in channel {channel_id}. Please specify them in your detector file.")

                adc_voltage_range = (
                    det_channel[field_prefix + "adc_min_voltage"] * units.V,
                    det_channel[field_prefix + "adc_max_voltage"] * units.V
                )
        else:
            if field_prefix + "adc_noise_nbits" in det_channel:
                error_msg = (
                    f"The field \"{field_prefix}adc_noise_nbits\" is present the detector description of channel "
                    f"{channel_id}. This field is deprecated. Please use the field \"{field_prefix}adc_noise_count\" "
                    "instead. To calculate the count form the nbits, use the formula: count = 2^(nbits - 1) - 1. "
                    "(This forumla is not intuitive which is why the old field was deprecated)."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            adc_noise_count_label = field_prefix + "adc_noise_count"
            if adc_noise_count_label not in det_channel:
                raise ValueError(
                    f"The field {adc_noise_count_label} is not present in channel {channel_id}. "
                    "Please specify it on your detector file.")

            adc_noise_count = det_channel[adc_noise_count_label]
            logger.debug(
                "Use a noise VRMS of {:.2f} mV and a ADC noise count of {} to define the ADC voltage range".format(
                    vrms / units.mV, adc_noise_count))
            adc_voltage_range_tmp = vrms * (2 ** adc_n_bits - 1) / adc_noise_count

            # make the assumption that the voltage range is symmetric around 0
            adc_voltage_range = (-adc_voltage_range_tmp / 2, adc_voltage_range_tmp / 2)

        logger.debug(
            ("ADC parameters: "
            "\n\tadc_voltage_range: ({}, {}) V"
            "\n\tadc_n_bits: {}"
            "\n\tadc_sampling_frequency: {} GHz"
            "\n\tadc_time_delay: {} ns").format(
                adc_voltage_range[0] / units.V, adc_voltage_range[1] / units.V,
                adc_n_bits, adc_sampling_frequency / units.GHz, adc_time_delay / units.ns
            ))

        return adc_n_bits, adc_voltage_range, adc_sampling_frequency, adc_time_delay

    def get_digital_trace(
        self, station, det, channel,
        Vrms=None,
        trigger_adc=False,
        clock_offset=0.0,
        adc_type='perfect_floor_comparator',
        return_sampling_frequency=False,
        adc_output='voltage',
        trigger_filter=None,
        adc_baseline_voltage=0):
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

        trigger_filter: array of floats
            Freq. domain of the response to be applied to post-ADC traces
            Must be length for "MC freq"
        adc_baseline_voltage: float (default: 0 V)
            The baseline voltage to be added to the trace before digitisation.

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
            det_channel, channel_id=channel_id, vrms=Vrms, trigger_adc=trigger_adc)

        if clock_offset:
            adc_time_delay += clock_offset / adc_sampling_frequency

        sampling_frequency = channel.get_sampling_rate()
        if adc_sampling_frequency > sampling_frequency:
            raise ValueError(
                'The ADC sampling rate is greater than '
                f'the channel {channel.get_id()} sampling rate. '
                'Please change the ADC sampling rate.')

        if trigger_filter is not None:
            apply_filter(channel, trigger_filter)

        if adc_time_delay:
            # Random clock offset
            trace, dt_tstart = signal_processing.delay_trace(channel, sampling_frequency, adc_time_delay)
            times = channel.get_times()
            if dt_tstart > 0:
                # by design dt_tstart is a multiple of the sampling rate
                times = times[int(round(dt_tstart / sampling_frequency)):]
            times = times[:len(trace)]
            channel.set_trace(trace, sampling_frequency, trace_start_time=times[0])

        # Add a baseline voltage to the trace
        if adc_baseline_voltage:
            logger.debug("Adding a baseline voltage of {:.2f} V to the trace".format(adc_baseline_voltage))
            channel.set_trace(channel.get_trace() + adc_baseline_voltage, "same")

        if adc_sampling_frequency != sampling_frequency:
            # Upsampling to 5 GHz before downsampling using interpolation.
            # We cannot downsample with a Fourier method because we want to keep
            # the higher Nyquist zones.
            upsampling_frequency = 5.0 * units.GHz
            if upsampling_frequency > sampling_frequency:
                channel.resample(upsampling_frequency)

            # Downsampling to ADC frequency
            resampled_times, resampled_trace = downsampling_linear_interpolation(
                channel.get_trace(), channel.get_sampling_rate(), adc_sampling_frequency)
            resampled_times += channel.get_trace_start_time()

            # Digitisation
            digital_trace = self._adc_types[adc_type](resampled_trace, adc_n_bits, adc_ref_voltage, adc_output)
        else:
            digital_trace = self._adc_types[adc_type](channel.get_trace(), adc_n_bits, adc_ref_voltage, adc_output)

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
            trigger_filter=None,
            adc_baseline_voltage=0):
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

        trigger_filter: array of floats
            Freq. domain of the response to be applied to post-ADC traces
            Must be length for "MC freq"
        adc_baseline_voltage: float (default: 0 V)
            The baseline voltage to be added to the trace before digitisation.
        """

        t = time.time()

        for channel in station.iter_channels():
            digital_trace, adc_sampling_frequency = self.get_digital_trace(
                station, det, channel,
                clock_offset=clock_offset,
                adc_type=adc_type,
                return_sampling_frequency=True,
                adc_output=adc_output,
                trigger_filter=trigger_filter,
                adc_baseline_voltage=adc_baseline_voltage
            )

            channel.set_trace(digital_trace, adc_sampling_frequency)

        self.__t += time.time() - t

    def end(self):
        pass


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
    interpolate_trace = scipy.interpolate.interp1d(
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
