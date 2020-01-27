import NuRadioReco.modules.channelGenericNoiseAdder
from NuRadioReco.utilities import units
import matplotlib.pyplot as plt
import numpy as np
from scipy import constants
import argparse
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
"""

parser = argparse.ArgumentParser(description='calculates noise trigger rate for phased array')
parser.add_argument('--ntries', type=int, help='number noise traces to which a trigger is applied for each threshold', default=1000)
parser.add_argument('--array', type=str, help='array type: RNO or ARA', default='RNO')
args = parser.parse_args()

main_low_angle = -53. * units.deg
main_high_angle = 47. * units.deg
default_angles = np.arcsin( np.linspace( np.sin(main_low_angle), np.sin(main_high_angle), 15) )

default_sec_angles = 0.5*(default_angles[:-1]+default_angles[1:])
default_sec_angles = np.insert(default_sec_angles, len(default_sec_angles), default_angles[-1] + 3.5*units.deg)

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

array_type = args.array

if (array_type == 'ARA'):
    min_freq = 130*units.MHz
    max_freq = 750*units.MHz
    sampling_rate = 1.5*units.GHz
    time_step = 1./sampling_rate
    window_width = 16
    only_primary = False
    primary_angles = default_angles
    secondary_angles = default_sec_angles
    ant_z_primary = [-46.5, -47.5, -48.5, -49.5, -50.5, -51.5, -52.5, -53.5] # primary antennas positions
    primary_channels = [0, 1, 2, 3, 4, 5, 6, 7] # channels used for primary beam
    beam_rolls = get_beam_rolls(ant_z_primary, primary_channels, primary_angles, time_step)
elif (array_type == 'RNO'):
    min_freq = 132*units.MHz
    max_freq = 700*units.MHz
    sampling_rate = 3*units.GHz
    time_step = 1./sampling_rate
    window_width = 32
    only_primary = True
    primary_angles = np.arcsin( np.linspace( np.sin(main_low_angle), np.sin(main_high_angle), 30) )
    ant_z_primary = [-98.5, -99.5, -100.5, -101.5] # primary antennas positions
    primary_channels = [0, 1, 2, 3] # channels used for primary beam
    beam_rolls = get_beam_rolls(ant_z_primary, primary_channels, primary_angles, time_step)

n_samples = 1000000 # number of samples
bandwidth = max_freq-min_freq
amplitude = (300 * 50 * constants.k * bandwidth / units.Hz) ** 0.5

Ntries = args.ntries # number of tries

threshold_factors = [2.15, 2.2, 2.25, 2.3, 2.35, 2.40]

if not only_primary:
    sec_channels = [0, 1, 3, 4, 6, 7] # channels used for secondary beam
    ant_z_secondary = [-46.5, -47.5, -49.5, -50.5, -52.5, -53.5] # secondary antennas positions
    dict_channels = {0:0, 1:1, 3:2, 4:3, 6:4, 7:5} # dictionary mapping primary channels to secondary channels
    sec_beam_rolls = get_beam_rolls(ant_z_secondary, sec_channels, secondary_angles, time_step)
else:
    sec_channels = []
    ant_z_secondary = []
    dict_channels = {}
    sec_beam_rolls = [{} for roll in beam_rolls]

n_beams = len(primary_angles)


for threshold_factor in threshold_factors:
    prob_per_window = 0
    for Ntry in range(Ntries):
        noise_array = []

        for iant in range(len(primary_channels)):

            noise_array.append(channelGenericNoiseAdder.bandlimited_noise(min_freq, max_freq, n_samples, sampling_rate, amplitude, type='rayleigh'))

        for subbeam_rolls, sec_subbeam_rolls in zip(beam_rolls, sec_beam_rolls):

            noise = None

            for channel_id in range(len(primary_channels)):

                if(noise is None):
                    noise = np.roll(noise_array[channel_id], subbeam_rolls[channel_id])
                else:
                    noise += np.roll(noise_array[channel_id], subbeam_rolls[channel_id])

                if channel_id in sec_channels:

                    sec_channel_id = dict_channels[channel_id]
                    noise += np.roll(noise_array[channel_id], sec_subbeam_rolls[sec_channel_id])

            n_windows = int(n_samples/window_width)
            n_windows = int(2*n_windows - 1)

            n_phased = len(primary_channels) + len(sec_channels)

            strides = noise.strides
            windowed_traces = np.lib.stride_tricks.as_strided(noise, \
                              shape=(n_windows, window_width), \
                              strides=(int(window_width / 2) * strides[0], strides[0]))

            squared_mean = np.sum(windowed_traces ** 2 / window_width, axis=1)
            squared_mean_threshold = n_phased * threshold_factor**2 * amplitude**2
            mask = squared_mean > squared_mean_threshold

            # If a phased direction triggers, the whole phased array triggers.
            # The following formula is justified as long as the probability is small
            # and each direction triggers independently of the rest.
            prob_per_window += np.sum( mask * np.ones(len(mask)) )/(n_windows*Ntries)

    # The 2 comes from the use of overlapping windows
    trigger_frequency = prob_per_window / (window_width/2)
    print('Threshold factor: {:.2f}, Fraction of noise triggers: {:.8f}%, Noise trigger rate: {:.2f}'.format(threshold_factor, prob_per_window*100., trigger_frequency/units.Hz))
