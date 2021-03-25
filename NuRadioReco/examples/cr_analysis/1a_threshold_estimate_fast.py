import scipy.constants
import numpy as np
import scipy.signal
import sys
import time
import pickle
import os
from itertools import count
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.channelGalacticNoiseAdder
import NuRadioReco.modules.trigger.envelopeTrigger
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.utilities.fft
import NuRadioReco.modules.eventTypeIdentifier
from NuRadioReco.detector.generic_detector import GenericDetector
from NuRadioReco.framework.base_trace import BaseTrace
from NuRadioReco.framework.event import Event
from NuRadioReco.framework.station import Station
from NuRadioReco.framework.channel import Channel
from NuRadioReco.utilities import units
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.modules.trigger.highLowThreshold import get_majority_logic
from NuRadioReco.framework.trigger import EnvelopeTrigger
import argparse
import pygdsm
import astropy

'''
The difference to 1_ is, that the envelope module is speed up.

This script calculates a first estimate from which the calculations of the threshold will continue. This is done by 
increasing the threshold after a number of iteration if more than one trigger triggered true. From the resulting 
threshold, the next script starts. So first run 1_threshold.py estimate and then use 2a_threshold_final_fast.py. 
Afterwards you have to use 3_threshold_average_and_plot.py to get a dictionary and plot with the results.

the sampling rate has a huge influence on the threshold, because the trace has more time to exceed the threshold
for a sampling rate of 1GHz, 1955034 iterations yields a resolution of 0.5 Hz
if galactic noise is used it adds a factor of 10 to the number of iterations because it dices the phase 10 times. this is done due to computation efficiency

For different passbands on the cluster I used something like this:

for low in $(seq 80 10 150)
do ((h=low+30))
echo $low
  for high in $(seq $h 10 400)
  do echo $high
  qsub /afs/ifh.de/group/radio/scratch/lpyras/Cluster_jobs/Cluster_ntr_1.sh 50000 $low $high
  sleep 0.1
  done
done
'''


parser = argparse.ArgumentParser()
parser.add_argument('output_path', type=os.path.abspath, nargs='?', default = '', help = 'Path to save output, most likely the path to the cr_analysis directory')
parser.add_argument('n_iterations', type=int, nargs='?', default = 20, help = 'number of iterations each threshold should be iterated over. Has to be a multiple of 10')
parser.add_argument('passband_low', type=int, nargs='?', default = 80, help = 'lower bound of the passband used for the trigger in MHz')
parser.add_argument('passband_high', type=int, nargs='?', default = 180, help = 'higher bound of the passband used for the trigger in MHz')
parser.add_argument('detector_file', type=str, nargs='?', default = 'LPDA_detector_southpole.json', help = 'detector file')
parser.add_argument('default_station', type=int, nargs='?', default = 101 , help = 'default station id')
parser.add_argument('sampling_rate', type=int, nargs='?', default = 1, help = 'sampling rate in GHz')
parser.add_argument('coinc_window', type=int, nargs='?', default = 60, help = 'coincidence window within the number coincidence has to occur. In ns')
parser.add_argument('number_coincidences', type=int, nargs='?', default = 2 , help = 'number coincidences of true trigger within on station of the detector')
parser.add_argument('order_trigger', type=int, nargs='?', default = 10, help = 'order of the filter used in the trigger')
parser.add_argument('Tnoise', type=int, nargs='?', default = 300 , help = 'Temperature of thermal noise in K')
parser.add_argument('T_noise_min_freq', type=int, nargs='?', default = 50, help = 'min freq of thermal noise in MHz')
parser.add_argument('T_noise_max_freq', type=int, nargs='?', default = 800, help = 'max freq of thermal noise in MHz')
parser.add_argument('galactic_noise_n_side', type=int, nargs='?', default = 4, help = 'The n_side parameter of the healpix map. Has to be power of 2, basicly the resolution')
parser.add_argument('galactic_noise_interpolation_frequencies_step', type=int, nargs='?', default = 100, help = 'frequency steps the galactic noise is interpolated over in MHz')
parser.add_argument('threshold_start', type=int, nargs='?', default = 0, help = 'value of the first tested threshold')
parser.add_argument('threshold_step', type=int, nargs='?', default = 0.00001, help = 'value of the threshold step')
parser.add_argument('station_time', type=str, nargs='?', default = '2019-01-01T00:00:00', help = 'station time for calculation of galactic noise')
parser.add_argument('station_time_random', type=bool, nargs='?', default = True, help = 'choose if the station time should be random or not')
parser.add_argument('hardware_response', type=bool, nargs='?', default = False, help = 'choose if the hardware response (amp) should be True or False')

args = parser.parse_args()
output_path = args.output_path
abs_output_path = os.path.abspath(args.output_path)
n_iterations = args.n_iterations / 10
n_iterations = int(n_iterations)
passband_low = args.passband_low
passband_high = args.passband_high
passband_trigger = np.array([passband_low, passband_high]) * units.megahertz
detector_file = args.detector_file
default_station = args.default_station
sampling_rate = args.sampling_rate * units.gigahertz
coinc_window = args.coinc_window * units.ns
number_coincidences = args.number_coincidences
order_trigger = args.order_trigger
Tnoise = args.Tnoise * units.kelvin
T_noise_min_freq = args.T_noise_min_freq * units.megahertz
T_noise_max_freq = args.T_noise_max_freq * units.megahertz
galactic_noise_n_side= args.galactic_noise_n_side
galactic_noise_interpolation_frequencies_step = args.galactic_noise_interpolation_frequencies_step
threshold_start = args.threshold_start * units.volt
threshold_step = args.threshold_step *units.volt
station_time = args.station_time
station_time_random = args.station_time_random
hardware_response = args.hardware_response

det = GenericDetector(json_filename=detector_file, default_station=default_station)

Vrms_thermal_noise = (((scipy.constants.Boltzmann * units.joule / units.kelvin) * Tnoise *
         (T_noise_max_freq - T_noise_min_freq ) * 50 * units.ohm)**0.5)
print('Vrms thermal Noise', Vrms_thermal_noise)

station_ids = det.get_station_ids()
station_id = station_ids[0]
channel_ids = det.get_channel_ids(station_id)

# set one empty event
event = Event(0, 0)
station = Station(station_id)
event.set_station(station)

station.set_station_time(astropy.time.Time(station_time))

if station_time_random == True:
    random_generator_hour = np.random.RandomState()
    hour = random_generator_hour.randint(0, 24)
    if hour < 10:
        station.set_station_time(astropy.time.Time('2019-01-01T0{}:00:00'.format(hour)))
    elif hour >= 10:
        station.set_station_time(astropy.time.Time('2019-01-01T{}:00:00'.format(hour)))

for channel_id in channel_ids: # take some channel id that match your detector
    channel = Channel(channel_id)
    default_trace = np.zeros(1024)
    channel.set_trace(trace=default_trace, sampling_rate=sampling_rate)
    station.add_channel(channel)

eventTypeIdentifier = NuRadioReco.modules.eventTypeIdentifier.eventTypeIdentifier()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
channelGenericNoiseAdder.begin()
channelGalacticNoiseAdder = NuRadioReco.modules.channelGalacticNoiseAdder.channelGalacticNoiseAdder()
channelGalacticNoiseAdder.begin(n_side=4, interpolation_frequencies=np.arange(10, 1100, galactic_noise_interpolation_frequencies_step) * units.MHz)
hardwareResponseIncorporator = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()

triggerSimulator = NuRadioReco.modules.trigger.envelopeTrigger.triggerSimulator()
triggerSimulator.begin()

t = time.time()  # absolute time of system
sampling_rate = station.get_channel(channel_ids[0]).get_sampling_rate()
dt = 1. / sampling_rate
# print('samling rate', sampling_rate)
# print('1/dt', dt)

time = channel.get_times()
channel_trace_start_time = time[0]
channel_trace_final_time = time[len(time)-1]
channel_trace_time_interval = channel_trace_final_time - channel_trace_start_time

trigger_status = []
triggered_trigger = []
trigger_rate = []
trigger_efficiency = []
thresholds = []
iterations = []

for n_thres in count():
    threshold = threshold_start + (n_thres * threshold_step)
    thresholds.append(threshold)
    trigger_status_per_all_it = []

    for n_it in range(n_iterations):
        print('iter', n_it)
        print(n_iterations)
        station = event.get_station(default_station)
        eventTypeIdentifier.run(event, station, "forced", 'cosmic_ray')
        for channel in station.iter_channels():
            default_trace = np.zeros(1024)
            channel.set_trace(trace=default_trace, sampling_rate=sampling_rate)

        channelGenericNoiseAdder.run(event, station, det, amplitude=Vrms_thermal_noise, min_freq=T_noise_min_freq, max_freq=T_noise_max_freq, type='rayleigh', bandwidth=None)
        channelGalacticNoiseAdder.run(event, station, det)
        if hardware_response == True:
            hardwareResponseIncorporator.run(event, station, det, sim_to_data=True)

        for i_phase in range(10):
            print('test iteration', (i_phase) + (n_it*10))
            for channel in station.iter_channels():
                # print('change phase in channel', channel.get_id())
                freq_specs = channel.get_frequency_spectrum()
                rand_phase = np.random.uniform(low=0, high= 2*np.pi, size=len(freq_specs))
                freq_specs = np.abs(freq_specs) * np.exp(1j * rand_phase)
                channel.set_frequency_spectrum(frequency_spectrum=freq_specs, sampling_rate=sampling_rate)

            trigger_status_one_it = []
            triggered_bins_channels = []
            channels_that_passed_trigger = []
            for channel in station.iter_channels():
                print('check trigger for channel', channel.get_id())
                frequencies = channel.get_frequencies()
                f = np.zeros_like(frequencies, dtype=np.complex)
                mask = frequencies > 0
                b, a = scipy.signal.butter(order_trigger, passband_trigger, 'bandpass', analog=True,
                                           output='ba')  # Numerator (b) and denominator (a) polynomials of the IIR filter
                w, h = scipy.signal.freqs(b, a, frequencies[
                    mask])
                f[mask] = h

                sampling_rate = channel.get_sampling_rate()
                freq_spectrum_fft_copy = np.array(channel.get_frequency_spectrum())
                freq_spectrum_fft_copy *= f
                trace_filtered = NuRadioReco.utilities.fft.freq2time(freq_spectrum_fft_copy, sampling_rate)

                triggered_bins = np.abs(scipy.signal.hilbert(trace_filtered)) > threshold
                print('trace_filtered > threshold', max(np.abs(scipy.signal.hilbert(trace_filtered))), threshold)
                triggered_bins_channels.append(triggered_bins)
                channel_rms = (np.std(trace_filtered)/units.V)
                channel_sigma = ((31.8 * units.microvolt) / (np.std(trace_filtered)/units.V))
                print(channel.get_id()-14, channel_rms, channel_sigma)


                if True in triggered_bins:
                    channels_that_passed_trigger.append(channel.get_id())
                #print('channel that passed trigger', channels_that_passed_trigger)

            # check for coincidences with get_majority_logic(tts, number_of_coincidences=,
            # time_coincidence= * units.ns, dt=1 * units.ns)
            # returns:
            # triggered: bool; returns True if majority logic is fulfilled --> has_triggered
            # triggered_bins: array of ints; the bins that fulfilled the trigger --> triggered_bins
            # triggered_times = triggered_bins * dt: array of floats;
            # the trigger times relative to the trace --> triggered_times
            has_triggered, triggered_bins, triggered_times = get_majority_logic(triggered_bins_channels,
                                                                                    number_coincidences, coinc_window, dt)

            trigger_status_one_it.append(has_triggered)
            #print('Trigger boolean current iteration', trigger_status_one_it)
            trigger_status_per_all_it.append(trigger_status_one_it)
            # print('Trigger all iteration', trigger_status_per_all_it)
            # print('true trigger', np.sum(trigger_status_per_all_it))
            # print('all trigger', len(trigger_status_per_all_it))
            # print('np.sum(trigger_status_per_all_it)', np.sum(trigger_status_per_all_it))

            #print('channel_rms',channel_rms)
            #print('channel_sigma', channel_sigma)


        if np.sum(trigger_status_per_all_it) > 1:
            trigger_efficiency_per_tt = np.sum(trigger_status_per_all_it) / len(trigger_status_per_all_it)
            trigger_rate_per_tt = (1 / channel_trace_time_interval) * trigger_efficiency_per_tt

            trigger_rate.append(trigger_rate_per_tt)
            trigger_efficiency.append(trigger_efficiency_per_tt)

            # print('true triggered', np.sum(trigger_status_per_all_it))
            # print('trigger efficiency of this threshold', trigger_efficiency_per_tt)
            # print('trigger rate of this threshold [Hz]', trigger_rate_per_tt / units.Hz)
            break;

        elif n_it == (n_iterations-1):
            trigger_efficiency_per_tt = np.sum(trigger_status_per_all_it) / len(trigger_status_per_all_it)
            trigger_rate_per_tt = (1 / channel_trace_time_interval) * trigger_efficiency_per_tt

            trigger_rate.append(trigger_rate_per_tt)
            trigger_efficiency.append(trigger_efficiency_per_tt)

            # print('true triggered', np.sum(trigger_status_per_all_it))
            # print('trigger efficiency of this threshold', trigger_efficiency_per_tt)
            # print('trigger rate of this threshold [Hz]', trigger_rate_per_tt / units.Hz)

            thresholds = np.array(thresholds)
            trigger_rate = np.array(trigger_rate)
            trigger_efficiency = np.array(trigger_efficiency)

            # print('thresholds tested', thresholds)
            # print('efficiency', trigger_efficiency)
            # print('trigger rate', trigger_rate)

            dic = {}
            dic['T_noise'] = Tnoise
            dic['Vrms_thermal_noise'] = Vrms_thermal_noise
            dic['thresholds'] = thresholds
            dic['efficiency'] = trigger_efficiency
            dic['trigger_rate'] = trigger_rate
            dic['n_iterations'] = n_iterations * 10 # from phase change in galactic noise
            dic['passband_trigger'] = passband_trigger
            dic['coinc_window'] = coinc_window
            dic['order_trigger'] = order_trigger
            dic['number_coincidences'] = number_coincidences
            dic['detector_file'] = detector_file
            dic['default_station'] = default_station
            dic['sampling_rate'] = sampling_rate
            dic['T_noise_min_freq'] = T_noise_min_freq
            dic['T_noise_max_freq '] = T_noise_max_freq
            dic['galactic_noise_n_side'] = galactic_noise_n_side
            dic['galactic_noise_interpolation_frequencies_step'] = galactic_noise_interpolation_frequencies_step
            dic['station_time'] = station_time
            dic['station_time_random'] = station_time_random
            dic['hardware_response'] = hardware_response

            print(dic)
            output_file = 'output_threshold_estimate/estimate_threshold_pb_{:.0f}_{:.0f}_i{}.pickle'.format(passband_trigger[0]/units.MHz,passband_trigger[1]/units.MHz, len(trigger_status_per_all_it))
            abs_path_output_file = os.path.normpath(os.path.join(abs_output_path, output_file))
            with open(abs_path_output_file,'wb') as pickle_out:
                pickle.dump(dic, pickle_out)

            sys.exit(0)

