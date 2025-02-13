import logging
import time
import numpy as np
from NuRadioReco.utilities import units
from scipy.interpolate import interp1d
import scipy.signal
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities.trace_utilities import delay_trace

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
    output: {'voltage', 'counts'}, default 'voltage'
        Options:

        * 'voltage' to store the ADC output as discretised voltage trace
        * 'counts' to store the ADC output in ADC counts

    Returns
    -------
    digital_trace: array of floats
        Digitised voltage trace in volts or ADC counts
    """

    lsb_voltage = adc_ref_voltage / (2 ** (adc_n_bits - 1) - 1)

    if (mode == 'floor'):
        digital_trace = np.floor(trace / lsb_voltage).astype(int)
    elif (mode == 'ceiling'):
        digital_trace = np.ceil(trace / lsb_voltage).astype(int)
    else:
        raise ValueError('Choose floor or ceiing as modes for the comparator ADC')

    digital_trace = apply_saturation(digital_trace, adc_n_bits, adc_ref_voltage)

    if (output == 'voltage'):
        digital_trace = lsb_voltage * digital_trace.astype(float)
    elif (output == 'counts'):
        pass
    else:
        raise ValueError("The ADC output format is unknown. Please choose 'voltage' or 'counts'")

    return digital_trace  # , lsb_voltage


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
        self._adc_types = {'perfect_floor_comparator': perfect_floor_comparator,
                           'perfect_ceiling_comparator': perfect_ceiling_comparator}
        self._mandatory_fields = ['adc_nbits',
                                  'adc_noise_nbits',
                                  'adc_sampling_frequency']

        self.logger = logging.getLogger('NuRadioReco.analogToDigitalConverter')

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

        field_prefix = ""
        if trigger_adc:
            field_prefix = "trigger_"
                              
        for field in self._mandatory_fields:
            field_check = field_prefix + field
            if field_check not in det_channel:
                raise ValueError(
                    f"The field {field_check} is not present in channel {channel_id}. "
                    "Please specify it on your detector file.")

        times = channel.get_times()
        trace = channel.get_trace()
        sampling_frequency = channel.get_sampling_rate()

        adc_time_delay_label = field_prefix + "adc_time_delay"
        adc_n_bits_label = field_prefix + "adc_nbits"
        adc_noise_n_bits = field_prefix + "adc_noise_nbits"
        adc_noise_n_bits_label = field_prefix + "adc_noise_nbits"
        adc_ref_voltage_label = field_prefix + "adc_reference_voltage"
        adc_sampling_frequency_label = field_prefix + "adc_sampling_frequency"

        # Use "or" if adc_time_delay_label is in dict but value is None
        adc_time_delay = (det_channel.get(adc_time_delay_label, 0) or 0) * units.ns

        adc_n_bits = det_channel[adc_n_bits_label]
        adc_noise_n_bits = det_channel[adc_noise_n_bits_label]
        adc_sampling_frequency = det_channel[adc_sampling_frequency_label] * units.GHz
        adc_time_delay += clock_offset / adc_sampling_frequency

        if Vrms is None:
            if adc_ref_voltage_label not in det_channel:
                raise ValueError(
                    f"The field {adc_ref_voltage_label} is not present in channel {channel_id}. "
                    "Please specify it on your detector file.")

            adc_ref_voltage = det_channel[adc_ref_voltage_label] * units.V
        else:
            adc_ref_voltage = Vrms * (2 ** (adc_n_bits - 1) - 1) / (2 ** (adc_noise_n_bits - 1) - 1)

        if adc_sampling_frequency > channel.get_sampling_rate():
            raise ValueError(
                'The ADC sampling rate is greater than '
                f'the channel {channel.get_id()} sampling rate. '
                'Please change the ADC sampling rate.')

        if trigger_filter is not None:

            trace_fft = np.fft.rfft(trace)
            if len(trace_fft) != len(trigger_filter):
                raise ValueError("Wrong filter length to apply to traces")

            trace = np.fft.irfft(trace_fft * trigger_filter)


        if adc_time_delay:
            # Random clock offset
            trace, dt_tstart = delay_trace(trace, sampling_frequency, adc_time_delay)
            if dt_tstart > 0:
                # by design dt_tstart is a multiple of the sampling rate
                times = times[int(round(dt_tstart / sampling_frequency)):]
            times = times[:len(trace)]

        # Upsampling to 5 GHz before downsampling using interpolation.
        # We cannot downsample with a Fourier method because we want to keep
        # the higher Nyquist zones.
        upsampling_frequency = 5.0 * units.GHz

        if upsampling_frequency > sampling_frequency:
            upsampling_nsamples = int(upsampling_frequency * len(trace) / sampling_frequency)
            perfectly_upsampled_trace, perfectly_upsampled_times = scipy.signal.resample(trace, upsampling_nsamples, t=times)

        else:
            perfectly_upsampled_trace = trace
            perfectly_upsampled_times = times

        interpolate_delayed_trace = interp1d(
            perfectly_upsampled_times, perfectly_upsampled_trace, kind='linear',
            fill_value=(perfectly_upsampled_trace[0], perfectly_upsampled_trace[-1]),
            bounds_error=False)

        # Downsampling to ADC frequency
        new_n_samples = int((adc_sampling_frequency / sampling_frequency) * len(trace))
        resampled_times = np.arange(new_n_samples) / adc_sampling_frequency
        resampled_times += channel.get_trace_start_time()
        resampled_trace = interpolate_delayed_trace(resampled_times)

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
