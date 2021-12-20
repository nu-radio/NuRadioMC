import numpy as np
import helper_cr_eff as hcr
import json
import os
from NuRadioReco.utilities import units
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str, nargs='?', default=os.path.dirname(__file__),
                    help='Path to save output, most likely the path to the cr_efficiency_analysis directory')
parser.add_argument('--detector_file', type=str, nargs='?',
                    default='LPDA_Southpole.json',
                    help='file with one antenna at and the geographic location, change triggered channels accordingly')
parser.add_argument('--target_global_trigger_rate', type=float, nargs='?', default=100,
                    help='trigger rate for all channels in Hz')
parser.add_argument('--trigger_name', type=str, nargs='?', default='high_low',
                    help='name of the trigger, high_low, envelope or power_integration')
parser.add_argument('--default_station', type=int, nargs='?', default=101,
                    help='default station id')
parser.add_argument('--trace_samples', type=int, nargs='?', default=1024,
                    help='elements in the array of one trace')
parser.add_argument('--sampling_rate', type=int, nargs='?', default=1,
                    help='sampling rate in GHz')
parser.add_argument('--triggered_channels', type=np.ndarray, nargs='?', default=np.array([1]),
                    help='channel on which the trigger is applied')
parser.add_argument('--total_number_triggered_channels', type=int, nargs='?', default=4,
                    help='number ot channels that trigger.')
parser.add_argument('--number_coincidences', type=int, nargs='?', default=2,
                    help='number coincidences of true trigger within on station of the detector')
parser.add_argument('--coinc_window', type=int, nargs='?', default=80,
                    help='coincidence window within the number coincidence has to occur. In ns')
parser.add_argument('--int_window', type=float, nargs='?', default=10,
                    help='integration time window [ns] for power_integration trigger')
parser.add_argument('--passband_low', type=int, nargs='?', default=80,
                    help='lower bound of the passband used for the trigger in MHz')
parser.add_argument('--passband_high', type=int, nargs='?', default=180,
                    help='higher bound of the passband used for the trigger in MHz')
parser.add_argument('--order_trigger', type=int, nargs='?', default=10,
                    help='order of the filter used in the trigger')
parser.add_argument('--Tnoise', type=int, nargs='?', default=300,
                    help='Temperature of thermal noise in K')
parser.add_argument('--T_noise_min_freq', type=int, nargs='?', default=50,
                    help='min freq of thermal noise in MHz')
parser.add_argument('--T_noise_max_freq', type=int, nargs='?', default=800,
                    help='max freq of thermal noise in MHz')
parser.add_argument('--galactic_noise_n_side', type=int, nargs='?', default=4,
                    help='The n_side parameter of the healpix map. Has to be power of 2, basicly the resolution')
parser.add_argument('--galactic_noise_interpolation_frequencies_start', type=int, nargs='?', default=10,
                    help='start frequency the galactic noise is interpolated over in MHz')
parser.add_argument('--galactic_noise_interpolation_frequencies_stop', type=int, nargs='?', default=1100,
                    help='stop frequency the galactic noise is interpolated over in MHz')
parser.add_argument('--galactic_noise_interpolation_frequencies_step', type=int, nargs='?', default=100,
                    help='frequency steps the galactic noise is interpolated over in MHz')
parser.add_argument('--n_random_phase', type=int, nargs='?', default=10,
                    help='for computing time reasons one galactic noise amplitude is reused '
                         'n_random_phase times, each time a random phase is added')
parser.add_argument('--threshold_start', type=float, nargs='?',
                    help='value of the first tested threshold in Volt')
parser.add_argument('--threshold_step', type=float, nargs='?',
                    help='value of the threshold step in Volt')
parser.add_argument('--station_time', type=str, nargs='?', default='2021-01-01T00:00:00',
                    help='station time for calculation of galactic noise')
parser.add_argument('--station_time_random', type=bool, nargs='?', default=True,
                    help='choose if the station time should be random or not')
parser.add_argument('--hardware_response', type=bool, nargs='?', default=True,
                    help='choose if the hardware response (amp) should be True or False')
parser.add_argument('--iterations_per_job', type=int, nargs='?', default=200,
                    help='choose if the hardware response (amp) should be True or False')
parser.add_argument('--number_of_allowed_trigger', type=bool, nargs='?', default=3,
                    help='The number of iterations is calculated to yield a trigger rate')

args = parser.parse_args()
target_global_trigger_rate = args.target_global_trigger_rate * units.Hz
passband_low = args.passband_low * units.megahertz
passband_high = args.passband_high * units.megahertz
passband_trigger = np.array([passband_low, passband_high])
sampling_rate = args.sampling_rate * units.gigahertz
coinc_window = args.coinc_window * units.ns
int_window = args.int_window * units.ns
Tnoise = args.Tnoise * units.kelvin
T_noise_min_freq = args.T_noise_min_freq * units.megahertz
T_noise_max_freq = args.T_noise_max_freq * units.megahertz
galactic_noise_interpolation_frequencies_start = args.galactic_noise_interpolation_frequencies_start * units.MHz
galactic_noise_interpolation_frequencies_stop = args.galactic_noise_interpolation_frequencies_stop * units.MHz
galactic_noise_interpolation_frequencies_step = args.galactic_noise_interpolation_frequencies_step * units.MHz

trace_length = args.trace_samples / sampling_rate
target_single_trigger_rate = hcr.get_single_channel_trigger_rate(
    target_global_trigger_rate, args.total_number_triggered_channels, args.number_coincidences, coinc_window)
n_iteration_for_one_allowed_trigger = (trace_length * target_single_trigger_rate) ** -1

n_iterations = int(n_iteration_for_one_allowed_trigger * args.number_of_allowed_trigger / args.n_random_phase)
resolution = (n_iteration_for_one_allowed_trigger * args.number_of_allowed_trigger * trace_length) ** -1

number_of_jobs = n_iterations / args.iterations_per_job
number_of_jobs = int(np.ceil(number_of_jobs))
Vrms_thermal_noise = hcr.calculate_thermal_noise_Vrms(Tnoise, T_noise_max_freq, T_noise_min_freq)

if args.threshold_start is None:
    if args.trigger_name == "power_integration":
        raise Exception('Please set threshold start value manually for the power integration trigger')
    else:
        if args.hardware_response:
            threshold_start = 1e3 * Vrms_thermal_noise
        elif not args.hardware_response:
            threshold_start = 1.8 * Vrms_thermal_noise
else:
    threshold_start = args.threshold_start * units.volt

if args.threshold_step is None:
    if args.trigger_name == "power_integration":
        raise Exception('Please set threshold step value manually for the power integration trigger')
    else:
        if args.hardware_response:
            threshold_step = 1e-3 * units.volt
        elif not args.hardware_response:
            threshold_step = 1e-6 * units.volt
else:
    threshold_step = args.threshold_step * units.volt

dic = {'T_noise': Tnoise, 'Vrms_thermal_noise': Vrms_thermal_noise, 'n_iterations_total': n_iterations,
       'number_of_allowed_trigger': args.number_of_allowed_trigger, 'iterations_per_job': args.iterations_per_job,
       'number_of_jobs': number_of_jobs, 'target_single_trigger_rate': target_single_trigger_rate,
       'target_global_trigger_rate': target_global_trigger_rate, 'resolution': resolution,
       'trigger_name': args.trigger_name, 'passband_trigger': passband_trigger,
       'total_number_triggered_channels': args.total_number_triggered_channels,
       'number_coincidences': args.number_coincidences, 'triggered_channels': args.triggered_channels,
       'coinc_window': coinc_window, 'int_window': int_window, 'order_trigger': args.order_trigger, 'detector_file': args.detector_file,
       'default_station': args.default_station, 'trace_samples': args.trace_samples, 'sampling_rate': sampling_rate,
       'trace_length': trace_length, 'T_noise_min_freq': T_noise_min_freq, 'T_noise_max_freq': T_noise_max_freq,
       'galactic_noise_n_side': args.galactic_noise_n_side,
       'galactic_noise_interpolation_frequencies_start': galactic_noise_interpolation_frequencies_start,
       'galactic_noise_interpolation_frequencies_stop': galactic_noise_interpolation_frequencies_stop,
       'galactic_noise_interpolation_frequencies_step': galactic_noise_interpolation_frequencies_step,
       'station_time': args.station_time, 'station_time_random': args.station_time_random,
       'hardware_response': args.hardware_response, 'n_random_phase': args.n_random_phase,
       'threshold_start': threshold_start, 'threshold_step': threshold_step}

os.makedirs(os.path.join(args.output_path, 'config/ntr'), exist_ok=True)

output_file = f'config/ntr/config_{args.trigger_name}_trigger_rate_{target_global_trigger_rate/units.Hz:.0f}Hz_coinc_{args.number_coincidences}of{args.total_number_triggered_channels}.json'

abs_path_output_file = os.path.normpath(os.path.join(args.output_path, output_file))

with open(abs_path_output_file, 'w') as outfile:
    json.dump(dic, outfile, cls=hcr.NumpyEncoder, indent=4, sort_keys=True)
