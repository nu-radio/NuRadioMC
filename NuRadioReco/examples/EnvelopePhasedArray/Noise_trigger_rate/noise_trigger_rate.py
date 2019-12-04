import NuRadioReco.modules.channelGenericNoiseAdder
from NuRadioReco.utilities import units
import matplotlib.pyplot as plt
import numpy as np
from scipy import constants
import argparse
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
from NuRadioReco.utilities.diodeSimulator import diodeSimulator
import NuRadioReco.framework.channel
from NuRadioReco.utilities.fft import time2freq, freq2time
from scipy.signal import butter, freqs

"""
This code calculates the probability per window for the whole phased
array (not just one beam) of triggering on noise, and the rate of
noise triggers.

The trigger is an envelope phased array trigger. According
to the ARA folks, the maximum rate should be around 10 Hz for the whole
phased array.

This code calculates the delays for the beams, introduces noise in all the
phased antennas, calculates the tunnel diode envelope and then phases the array
into the directions given by default_angles. It verifies if there has been a
trigger and repeats.

Increasing the number of samples is equivalent to increasing the number of
tries, since the code calculates the probability of randomly triggering a
single window filled with noise.
"""

parser = argparse.ArgumentParser(description='calculates noise trigger rate for phased array')
parser.add_argument('--ntries', type=int, help='number noise traces to which a trigger is applied for each threshold', default=100)
args = parser.parse_args()

main_low_angle = -50. * units.deg
main_high_angle = 50. * units.deg
default_angles = np.arcsin( np.linspace( np.sin(main_low_angle), np.sin(main_high_angle), 15) )

def get_beam_rolls(ant_z, channel_list, phasing_angles=default_angles, time_step=2./3*units.ns, ref_index=1.78):
    """
    Calculates the delays needed for phasing the array.
    """

    beam_rolls = []
    ref_z = (np.max(ant_z)+np.min(ant_z))/2

    n_phased = len(channel_list)

    for angle in phasing_angles:
        subbeam_rolls = {}
        for z, channel_id in zip(ant_z, range(n_phased)):
            delay = (z-ref_z)/0.3 * ref_index * np.sin(angle)
            roll = int(delay/time_step)
            subbeam_rolls[channel_id] = roll
        beam_rolls.append(subbeam_rolls)

        return beam_rolls

min_freq = 132*units.MHz
max_freq = 700*units.MHz
sampling_rate = 3*units.GHz
window_width = 32

primary_angles = np.arcsin( np.linspace( np.sin(main_low_angle), np.sin(main_high_angle), 30) )

n_samples = 1000000 # number of samples
time_step = 1./sampling_rate
bandwidth = max_freq-min_freq
amplitude = (300 * 50 * constants.k * bandwidth / units.Hz) ** 0.5

Ntries = args.ntries # number of tries

threshold_factors = [3, 3.25, 3.5]

ratios = []

ant_z_primary = [-98.5, -99.5, -100.5, -101.5] # primary antennas positions
primary_channels = [0, 1, 2, 3] # channels used for primary beam
beam_rolls = get_beam_rolls(ant_z_primary, primary_channels, primary_angles, time_step)

n_beams = len(primary_angles)

passband = (100*units.MHz, 200*units.MHz)
diode = diodeSimulator(passband)
power_mean, power_std = diode.calculate_noise_parameters(sampling_rate,
                                                         min_freq,
                                                         max_freq,
                                                         amplitude=amplitude)

for threshold_factor in threshold_factors:
    prob_cross = 0

    n_phased = len(primary_channels)
    low_trigger  = power_mean - power_std * np.abs(threshold_factor)
    low_trigger *= n_phased

    for Ntry in range(Ntries):
        noise_array = []

        for iant in range(len(primary_channels)):

            noise_trace = channelGenericNoiseAdder.bandlimited_noise(min_freq, max_freq, n_samples,
                                                sampling_rate, amplitude, type='rayleigh')
            channel = NuRadioReco.framework.channel.Channel(0)
            channel.set_trace(noise_trace, sampling_rate)

            enveloped_noise = diode.tunnel_diode(channel)

            noise_array.append(enveloped_noise)

        for subbeam_rolls in beam_rolls:

            noise = None

            for channel_id in range(len(primary_channels)):

                if(noise is None):
                    noise = np.roll(noise_array[channel_id], subbeam_rolls[channel_id])
                else:
                    noise += np.roll(noise_array[channel_id], subbeam_rolls[channel_id])

            n_windows = int(n_samples/window_width)
            n_windows = int(2*n_windows - 1)

            threshold_passed = [ np.any(np.lib.stride_tricks.as_strided(noise[int(i_window*window_width/2):],(window_width,)) < low_trigger) \
                                 for i_window in np.linspace(0,n_windows-1,n_windows) ]
            threshold_passed = np.array(threshold_passed)

            # If a phased direction triggers, the whole phased array triggers.
            # That is why we multiply the probability by the number of beams, or phasing directions.
            # This is justified as long as the probability is small and each direction triggers
            # independently of the rest.
            prob_cross += np.sum( threshold_passed )/n_windows * n_beams

    ratio = float(prob_cross)/Ntries
    trigger_frequency = 2*ratio/window_width / units.Hz
    print('Threshold factor: {:.2f}, Fraction of noise triggers: {:.5f}%, Noise trigger rate: {:.2f} Hz'.format(threshold_factor, ratio*100., trigger_frequency))

    ratios.append(ratio)
