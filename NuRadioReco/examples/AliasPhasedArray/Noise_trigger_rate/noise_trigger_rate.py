import NuRadioReco.modules.channelGenericNoiseAdder
from NuRadioReco.utilities import units
import numpy as np
from scipy import constants
import argparse
from NuRadioReco.utilities.trace_utilities import butterworth_filter_trace, upsampling_fir
from NuRadioReco.modules.analogToDigitalConverter import perfect_floor_comparator, perfect_ceiling_comparator
from scipy.interpolate import interp1d
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()

"""
This code calculates the probability per window for the whole phased
array (not just one beam) of triggering on noise, and the rate of
noise triggers.

The trigger is an ARA-like power trigger and a phased array. According
to the ARA folks, the maximum rate should be around 10 Hz for the whole
phased array.

This code calculates the delays for the primary and secondary beams, couples
a primary to a secondary, introduces noise in all the phased antennas and then
phases the array into the directions given by default_angles and default_sec_angles.
It verifies if there has been a trigger and repeats.

Increasing the number of samples is equivalent to increasing the number of
tries, since the code calculates the probability of randomly triggering a
single window filled with noise.

This version of the code is meant to calculate the noise trigger rate using
analog-to-digital converters and different Nyquist zones (alias). The
different Nyquist zones present different different kinds of noise, as it
can be expected.
"""

parser = argparse.ArgumentParser(description='calculates noise trigger rate for phased array')
parser.add_argument('--ntries', type=int, help='number noise traces to which a trigger is applied for each threshold', default=1000)
parser.add_argument('--nyquist_zone', type=int, help='Nyquist zone', default=1)
parser.add_argument('--upsampling_factor', type=int, help='Upsampling factor (integer)', default=4)
parser.add_argument('--adc_sampling_frequency', type=float, help='Sampling frequency in GHz', default=250 * units.MHz)
parser.add_argument('--noise_rms_bits', type=float, help='Bits reserved for the noise RMS', default=2)
parser.add_argument('--adc_n_bits', type=int, help='ADC number of bits', default=8)
parser.add_argument('--threshold_factor', type=float, help='Threshold factor', default=8)
args = parser.parse_args()

main_low_angle = -50. * units.deg
main_high_angle = 50. * units.deg
default_angles = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 30))


def get_beam_rolls(
        ant_z,
        channel_list,
        phasing_angles=default_angles,
        time_step=2. / 3 * units.ns,
        ref_index=1.75,
        upsampling_factor=1):
    """
    Calculates the delays needed for phasing the array.
    """

    beam_rolls = []
    ref_z = (np.max(ant_z) + np.min(ant_z)) / 2

    n_phased = len(channel_list)

    for angle in phasing_angles:
        subbeam_rolls = {}
        for z, channel_id in zip(ant_z, range(n_phased)):
            delay = (z - ref_z) / 0.3 * ref_index * np.sin(angle)
            roll = int(np.round(delay * upsampling_factor / time_step))
            subbeam_rolls[channel_id] = roll
        beam_rolls.append(subbeam_rolls)

    return beam_rolls


def get_noise_rms_nyquist_zone(
        trace,
        input_sampling_frequency,
        adc_sampling_frequency,
        nyquist_zone=2,
        adc_n_bits=8,
        noise_rms_bits=2,
        bandwidth_edge=20 * units.MHz,
        min_freq=120 * units.MHz):
    """
    Calculates the noise RMS in one of the Nyquist zones for the ADC

    Parameters
    ----------
    trace: array of floats
        Noise trace
    input_sampling_frequency: float
        Sampling frequency of the input trace
    adc_sampling_frequency: float
        ADC sampling frequency
    nyquist_zone: integer
        Nyquist zone number
    adc_n_bits: integer
        ADC number of bits
    noise_rms_bits: float
        Bits reserved for the noise RMS
    bandwidth_edge: float
        Frequency interval used for filtering the chosen Nyquist zone.
        See above
    min_freq: float
        Lower frequency for the first Nyquist zone

    Returns
    -------
    noise_rms: float
        Standard deviation of the trace filtered in the chosen Nyquist zone
    """

    passband = ((nyquist_zone - 1) * adc_sampling_frequency / 2 + bandwidth_edge,
                nyquist_zone * adc_sampling_frequency / 2 - bandwidth_edge)

    if nyquist_zone == 1:
        passband = (min_freq, passband[1])

    filtered_trace = butterworth_filter_trace(trace, input_sampling_frequency,
                                              passband)

    noise_rms_previous = 0
    noise_rms = np.std(filtered_trace)
    adc_ref_voltage = get_ref_voltage(noise_rms, adc_n_bits, noise_rms_bits=noise_rms_bits)

    while(np.abs(noise_rms_previous - noise_rms) > 5e-8):

        noise_rms_previous = noise_rms
        digital_trace = get_digital_trace(
            trace,
            input_sampling_frequency,
            adc_sampling_frequency,
            adc_ref_voltage=adc_ref_voltage,
            upsampling_factor=upsampling_factor,
            nyquist_zone=nyquist_zone)
        noise_rms = np.std(digital_trace)
        adc_ref_voltage = get_ref_voltage(noise_rms, adc_n_bits, noise_rms_bits=noise_rms_bits)

    return noise_rms


def get_ref_voltage(noise_rms,
                    adc_n_bits,
                    noise_rms_bits=2):

    ref_voltage = noise_rms * (2**(adc_n_bits - 1) - 1) / 2**(noise_rms_bits - 1)

    return ref_voltage


def get_digital_trace(
        trace,
        input_sampling_frequency,
        adc_sampling_frequency,
        adc_n_bits=8,
        adc_ref_voltage=0.7 * units.mV,
        random_clock_offset=True,
        adc_type='perfect_floor_comparator',
        output='voltage',
        upsampling_factor=None,
        nyquist_zone=2,
        bandwidth_edge=20 * units.MHz,
        min_freq=120 * units.MHz):
    """
    Returns the trace converted by an ADC.

    Parameters
    ----------
    trace: array of floats
        Voltage trace to be digitised
    input_sampling_frequency: float
        Sampling frequency of the input
    adc_sampling_frequency: float
        Sampling frequency of the ADC
    adc_n_bits: integer
        ADC number of bits
    adc_ref_voltage
    random_clock_offset: bool
        If True, a random clock offset between -1 and 1 clock cycles is added
    adc_type: string
        The type of ADC used. The following are available:
        - perfect_floor_comparator
        - perfect_ceiling_comparator
        See functions with the same name on this module for documentation
    output: string
        - 'voltage' to store the ADC output as discretised voltage trace
        - 'counts' to store the ADC output in ADC counts
    upsampling_factor: integer
        Upsampling factor. The digital trace will be a upsampled to a
        sampling frequency int_factor times higher than the original one
    nyquist_zone: integer
        If None, the trace is not filtered
        If n, it uses the n-th Nyquist zone by applying an 8th-order
        Butterworth filter with critical frequencies:
        (n-1) * adc_sampling_frequency/2 + bandwidth_edge
        and
        n * adc_sampling_frequency/2 - bandwidth_edge
    bandwidth_edge: float
        Frequency interval used for filtering the chosen Nyquist zone.
        See above
    min_freq: float
        Lower frequency for the first Nyquist zone

    Returns
    -------
    digital_trace: array of floats
        Digitised voltage trace
    adc_sampling_frequency: float
        ADC sampling frequency for the channel
    """

    times = np.arange(len(trace)) / input_sampling_frequency

    adc_time_delay = 0

    if random_clock_offset:
        clock_offset = np.random.uniform(-1, 1)
        adc_time_delay += clock_offset / adc_sampling_frequency

    # Choosing Nyquist zone
    if nyquist_zone is not None:

        if nyquist_zone < 1:
            error_msg = "Nyquist zone is less than one. Exiting."
            raise ValueError(error_msg)
        if not isinstance(nyquist_zone, int):
            try:
                nyquist_zone = int(nyquist_zone)
            except:
                error_msg = "Could not convert nyquist_zone to integer. Exiting."
                raise ValueError(error_msg)

        passband = ((nyquist_zone - 1) * adc_sampling_frequency / 2 + bandwidth_edge,
                    nyquist_zone * adc_sampling_frequency / 2 - bandwidth_edge)
        if nyquist_zone == 1:
            passband = (min_freq, passband[1])

        filtered_trace = butterworth_filter_trace(trace, input_sampling_frequency,
                                                  passband)

    # Random clock offset
    delayed_times = times + adc_time_delay
    interpolate_trace = interp1d(times, filtered_trace, kind='linear',
                                 fill_value=(trace[0], trace[-1]),
                                 bounds_error=False)

    delayed_trace = interpolate_trace(delayed_times)

    interpolate_delayed_trace = interp1d(times, delayed_trace, kind='linear',
                                         fill_value=(delayed_trace[0], delayed_trace[-1]),
                                         bounds_error=False)

    # Downsampling to ADC frequency
    new_n_samples = int((adc_sampling_frequency / input_sampling_frequency) * len(delayed_trace))
    resampled_times = np.linspace(0, new_n_samples / adc_sampling_frequency, new_n_samples)
    resampled_trace = interpolate_delayed_trace(resampled_times)

    # Digitisation
    if adc_type == 'perfect_floor_comparator':
        digital_trace = perfect_floor_comparator(resampled_trace, adc_n_bits,
                                                 adc_ref_voltage, output)
    elif adc_type == 'perfect_ceiling_comparator':
        digital_trace = perfect_ceiling_comparator(resampled_trace, adc_n_bits,
                                                   adc_ref_voltage, output)

    # Upsampling with an FIR filter (if desired)
    if upsampling_factor is not None:
        if (upsampling_factor >= 2):
            upsampling_factor = int(upsampling_factor)
            upsampled_trace = upsampling_fir(digital_trace, adc_sampling_frequency,
                                             int_factor=upsampling_factor, ntaps=2**4)
            adc_sampling_frequency *= upsampling_factor

            digital_trace = upsampled_trace[:]

    # Ensuring trace has an even number of samples
    if (len(digital_trace) % 2 == 1):
        digital_trace = digital_trace[:-1]

    return digital_trace


nyquist_zone = args.nyquist_zone
upsampling_factor = args.upsampling_factor
adc_sampling_frequency = args.adc_sampling_frequency
noise_rms_bits = args.noise_rms_bits
adc_n_bits = args.adc_n_bits
threshold_factor = args.threshold_factor
Ntries = args.ntries    # number of tries

input_sampling_frequency = 5 * units.GHz
min_freq = 132 * units.MHz
max_freq = 700 * units.MHz
input_time_step = 1 / adc_sampling_frequency
time_step = input_time_step / upsampling_factor
window_time = 12 * units.ns
window_width = int(window_time / time_step)
primary_angles = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 30))
ant_z_primary = [-98.5, -99.5, -100.5, -101.5]  # primary antennas positions
primary_channels = [0, 1, 2, 3]     # channels used for primary beam
beam_rolls = get_beam_rolls(ant_z_primary, primary_channels, primary_angles,
                            input_time_step, upsampling_factor=upsampling_factor)

n_samples = 1000000     # number of samples
adc_n_samples = int(n_samples * adc_sampling_frequency * upsampling_factor / input_sampling_frequency)
bandwidth = max_freq - min_freq
amplitude = (300 * 50 * constants.k * bandwidth / units.Hz) ** 0.5

threshold_factors = [threshold_factor]

noise_trace = channelGenericNoiseAdder.bandlimited_noise(min_freq, max_freq, 5 * n_samples,
                                                         input_sampling_frequency, amplitude,
                                                         type='rayleigh')

noise_rms = get_noise_rms_nyquist_zone(noise_trace,
                                       input_sampling_frequency,
                                       adc_sampling_frequency,
                                       nyquist_zone=nyquist_zone,
                                       adc_n_bits=adc_n_bits,
                                       noise_rms_bits=noise_rms_bits,
                                       bandwidth_edge=20 * units.MHz,
                                       min_freq=min_freq)
adc_ref_voltage = get_ref_voltage(noise_rms, adc_n_bits=adc_n_bits, noise_rms_bits=noise_rms_bits)

print("Number of bits for noise RMS: {:.1f}".format(noise_rms_bits))
print("Number of bits of the ADC: {:d}".format(adc_n_bits))
print("Nyquist zone number {:d}".format(nyquist_zone))
print("Reference voltage: {:.3e} V".format(adc_ref_voltage / units.V))
print("Noise for the Nyquist zone: {:.3e} V".format(noise_rms / units.V))

n_beams = len(primary_angles)


for threshold_factor in threshold_factors:

    threshold_voltage = threshold_factor * noise_rms    # noise_rms
    prob_per_window = 0
    for Ntry in range(Ntries):
        noise_array = []

        for iant in range(len(primary_channels)):

            analog_noise = channelGenericNoiseAdder.bandlimited_noise(min_freq, max_freq, n_samples, input_sampling_frequency, amplitude, type='rayleigh')
            digital_noise = get_digital_trace(analog_noise,
                                              input_sampling_frequency,
                                              adc_sampling_frequency,
                                              adc_ref_voltage=adc_ref_voltage,
                                              upsampling_factor=upsampling_factor,
                                              nyquist_zone=nyquist_zone)
            noise_array.append(digital_noise)

        for subbeam_rolls in beam_rolls:

            noise = None

            for channel_id in range(len(primary_channels)):

                if(noise is None):
                    noise = np.roll(noise_array[channel_id], subbeam_rolls[channel_id])
                else:
                    noise += np.roll(noise_array[channel_id], subbeam_rolls[channel_id])

            n_windows = int(len(noise) / window_width)
            n_windows = int(2 * n_windows - 1)

            n_phased = len(primary_channels)

            strides = noise.strides
            windowed_traces = np.lib.stride_tricks.as_strided(
                noise,
                shape=(n_windows, window_width),
                strides=(int(window_width / 2) * strides[0], strides[0]),
                writeable=False
            )

            squared_mean = np.sum(windowed_traces ** 2 / window_width, axis=1)
            squared_mean_threshold = n_phased * (threshold_voltage)**2
            mask = squared_mean > squared_mean_threshold

            # If a phased direction triggers, the whole phased array triggers.
            # The following formula is justified as long as the probability is small
            # and each direction triggers independently of the rest.
            prob_per_window += np.sum(mask * np.ones(len(mask))) / (n_windows * Ntries)

    # The 2 comes from the use of overlapping windows
    trigger_frequency = prob_per_window / (window_time / 2)

    print('Threshold voltage: {:.3e} V, Fraction of noise triggers: {:.8f}%, Noise trigger rate: {:.2f} Hz'.format(threshold_voltage, prob_per_window * 100., trigger_frequency / units.Hz))
