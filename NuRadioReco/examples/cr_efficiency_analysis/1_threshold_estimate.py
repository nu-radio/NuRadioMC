import numpy as np
import helper_cr_eff as hcr
import time
import json
import os
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.channelGalacticNoiseAdder
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.trigger.envelopeTrigger
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.eventTypeIdentifier
from NuRadioReco.detector.generic_detector import GenericDetector
from NuRadioReco.utilities import units
from NuRadioReco.modules.trigger.highLowThreshold import get_majority_logic
import NuRadioReco.modules.trigger.highLowThreshold
from NuRadioReco.framework.trigger import EnvelopeTrigger
import argparse
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Threshold estimate')

'''
The difference to 1_threshold_estimate.py is, that the trigger module of the envelope trigger 
is directly implemented and speeded up.
If you used the 1_.._fast.py, please use 2_..._fast.py

This script calculates a first estimate from which the calculations of the threshold will continue. This is done by 
increasing the threshold after a number of iteration if more than one trigger triggered true. From the resulting 
threshold, the next script starts. So first run 1_threshold_estimate.py estimate and then use 2_threshold_final.py. 
Afterwards you have to use 3_create_one_dict_and_plot.py to get a dictionary and plot with the results.

For the galactic noise, the sky maps from PyGDSM are used. You can install it with 
pip install git+https://github.com/telegraphic/pygdsm .

The sampling rate has a huge influence on the threshold, because the trace has more time to exceed the threshold
for a sampling rate of 1GHz, 1955034 iterations yields a resolution of 0.5 Hz
if galactic noise is used it adds a factor of 10 (n_random_phase) to the number of iterations because it dices the phase 10 times. 
This is done due to computation efficiency

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
parser.add_argument('output_path', type=os.path.abspath, nargs='?', default='',
                    help='Path to save output, most likely the path to the cr_efficiency_analysis directory')
parser.add_argument('n_iterations', type=int, nargs='?', default=10,
                    help='number of iterations each threshold should be iterated over. '
                         'Has to be a multiple of 10 (n_random_phase)')
parser.add_argument('number_of_allowed_trigger', type=int, nargs='?', default=10,
                    help='number of allowed_trigger out of the iteration number')
parser.add_argument('trigger_name', type=str, nargs='?', default='high_low',
                    help='name of the trigger, high_low or envelope')
parser.add_argument('passband_low', type=int, nargs='?', default=80,
                    help='lower bound of the passband used for the trigger in MHz')
parser.add_argument('passband_high', type=int, nargs='?', default=180,
                    help='higher bound of the passband used for the trigger in MHz')
parser.add_argument('detector_file', type=str, nargs='?', default='../example_data/arianna_station_32.json',
                    help='detector file, change triggered channels accordingly')
parser.add_argument('triggered_channels', type=np.ndarray, nargs='?', default=np.array([1,2,3,4]),
                    help='channel on which the trigger is applied')
parser.add_argument('default_station', type=int, nargs='?', default=32,
                    help='default station id')
parser.add_argument('sampling_rate', type=int, nargs='?', default=1,
                    help='sampling rate in GHz')
parser.add_argument('coinc_window', type=int, nargs='?', default=80,
                    help='coincidence window within the number coincidence has to occur. In ns')
parser.add_argument('number_coincidences', type=int, nargs='?', default=1,
                    help='number coincidences of true trigger within on station of the detector')
parser.add_argument('order_trigger', type=int, nargs='?', default=10,
                    help='order of the filter used in the trigger')
parser.add_argument('Tnoise', type=int, nargs='?', default=300,
                    help='Temperature of thermal noise in K')
parser.add_argument('T_noise_min_freq', type=int, nargs='?', default=50,
                    help='min freq of thermal noise in MHz')
parser.add_argument('T_noise_max_freq', type=int, nargs='?', default=800,
                    help='max freq of thermal noise in MHz')
parser.add_argument('galactic_noise_n_side', type=int, nargs='?', default=4,
                    help='The n_side parameter of the healpix map. Has to be power of 2, basicly the resolution')
parser.add_argument('galactic_noise_interpolation_frequencies_start', type=int, nargs='?', default=10,
                    help='start frequency the galactic noise is interpolated over in MHz')
parser.add_argument('galactic_noise_interpolation_frequencies_stop', type=int, nargs='?', default=1100,
                    help='stop frequency the galactic noise is interpolated over in MHz')
parser.add_argument('galactic_noise_interpolation_frequencies_step', type=int, nargs='?', default=100,
                    help='frequency steps the galactic noise is interpolated over in MHz')
parser.add_argument('n_random_phase', type=int, nargs='?', default=10,
                    help='for computing time reasons one galactic noise amplitude is reused '
                         'n_random_phase times, each time a random phase is added')
parser.add_argument('threshold_start', type=int, nargs='?', default=0,
                    help='value of the first tested threshold in Volt')
parser.add_argument('threshold_step', type=int, nargs='?', default=1e-6,
                    help='value of the threshold step in Volt')
parser.add_argument('station_time', type=str, nargs='?', default='2019-01-01T00:00:00',
                    help='station time for calculation of galactic noise')
parser.add_argument('station_time_random', type=bool, nargs='?', default=True,
                    help='choose if the station time should be random or not')
parser.add_argument('hardware_response', type=bool, nargs='?', default=False,
                    help='choose if the hardware response (amp) should be True or False')
args = parser.parse_args()
output_path = args.output_path
n_random_phase = args.n_random_phase
abs_output_path = os.path.abspath(args.output_path)
n_iterations = args.n_iterations / n_random_phase
n_iterations = int(n_iterations)
number_of_allowed_trigger = args.number_of_allowed_trigger
trigger_name = args.trigger_name
passband_low = args.passband_low * units.megahertz
passband_high = args.passband_high * units.megahertz
passband_trigger = np.array([passband_low, passband_high])
detector_file = args.detector_file
triggered_channels = args.triggered_channels
default_station = args.default_station
sampling_rate = args.sampling_rate * units.gigahertz
coinc_window = args.coinc_window * units.ns
number_coincidences = args.number_coincidences
order_trigger = args.order_trigger
Tnoise = args.Tnoise * units.kelvin
T_noise_min_freq = args.T_noise_min_freq * units.megahertz
T_noise_max_freq = args.T_noise_max_freq * units.megahertz
galactic_noise_n_side = args.galactic_noise_n_side
galactic_noise_interpolation_frequencies_start = args.galactic_noise_interpolation_frequencies_start * units.MHz
galactic_noise_interpolation_frequencies_stop = args.galactic_noise_interpolation_frequencies_stop * units.MHz
galactic_noise_interpolation_frequencies_step = args.galactic_noise_interpolation_frequencies_step * units.MHz
threshold_start = args.threshold_start * units.volt
threshold_step = args.threshold_step * units.volt
station_time = args.station_time
station_time_random = args.station_time_random
hardware_response = args.hardware_response

det = GenericDetector(json_filename=detector_file, default_station=default_station)
station_ids = det.get_station_ids()
channel_ids = det.get_channel_ids(station_ids[0])

logger.info("Apply {} trigger".format(trigger_name))
logger.info("Apply passband {} ".format(passband_trigger / units.MHz))

# The thermal noise for the ChannelGenericNoiseAdder is calculated here with a given temperature
Vrms_thermal_noise = hcr.calculate_thermal_noise_Vrms(Tnoise, T_noise_max_freq, T_noise_min_freq)
logger.info("Vrms thermal noise is {} V".format(Vrms_thermal_noise))

event, station, channel = hcr.create_empty_event(det, station_time, station_time_random, sampling_rate)

eventTypeIdentifier = NuRadioReco.modules.eventTypeIdentifier.eventTypeIdentifier()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
channelGenericNoiseAdder.begin()
channelGalacticNoiseAdder = NuRadioReco.modules.channelGalacticNoiseAdder.channelGalacticNoiseAdder()
channelGalacticNoiseAdder.begin(n_side=galactic_noise_n_side,
            interpolation_frequencies=np.arange(galactic_noise_interpolation_frequencies_start,
                                                galactic_noise_interpolation_frequencies_stop,
                                                galactic_noise_interpolation_frequencies_step))
hardwareResponseIncorporator = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelBandPassFilter.begin()

if trigger_name == 'high_low':
    triggerSimulator = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
    triggerSimulator.begin()

if trigger_name == 'envelope':
    triggerSimulator = NuRadioReco.modules.trigger.envelopeTrigger.triggerSimulator()
    triggerSimulator.begin()

t = time.time()  # absolute time of system
sampling_rate = station.get_channel(channel_ids[0]).get_sampling_rate()
dt = 1. / sampling_rate

time = station.get_channel(channel_ids[0]).get_times()
channel_trace_start_time = time[0]
channel_trace_final_time = time[len(time)-1]
channel_trace_time_interval = channel_trace_final_time - channel_trace_start_time

trigger_status = []
triggered_trigger = []
trigger_rate = []
trigger_efficiency = []
thresholds = []
iterations = []
channel_rms = []
channel_sigma = []

# with each iteration the threshold increases one step
n_thres = 0
number_of_trigger = number_of_allowed_trigger + n_random_phase
while number_of_trigger > number_of_allowed_trigger:
    n_thres += 1
    threshold = threshold_start + (n_thres * threshold_step)
    thresholds.append(threshold)
    logger.info("Processing threshold {}".format(threshold))
    trigger_status_per_all_it = []

    # here is number of iteration you want to check on (iteration is just a proxy for the time interval
    # on which you allow a certain number of trigger. In this case is every iteration 1024 ns (tracelength)
    # long and one trigger is allowed)
    for n_it in range(n_iterations):
        station = event.get_station(default_station)
        eventTypeIdentifier.run(event, station, "forced", 'cosmic_ray')

        #here an empty channel trace is created
        channel = hcr.create_empty_channel_trace(station, sampling_rate)

        # thermal and galactic noise is added
        channelGenericNoiseAdder.run(event, station, det, amplitude=Vrms_thermal_noise, min_freq=T_noise_min_freq,
                                     max_freq=T_noise_max_freq, type='rayleigh')
        channelGalacticNoiseAdder.run(event, station, det)

        # includes the amplifier response, if set true at the beginning
        if hardware_response == True:
            hardwareResponseIncorporator.run(event, station, det, sim_to_data=True)
        #print(np.sqrt(np.mean((station.get_channel(13).get_trace()/units.mV)**2)))
        # This loop changes the phase of a trace with rand_phase, this is because the GalacticNoiseAdder
        # needs some time and one amplitude is good enough for several traces.
        # The current number of iteration can be calculated with i_phase + n_it*n_random_phase
        for i_phase in range(n_random_phase):
            # attention: this loop will give you a warning:
            # WARNING:BaseStation:station has already a trigger with name high_low. The previous trigger will be overridden!
            # this is okay, because the trigger boolean (true/false) will be stored directly and can be overriden afterwards.
            trigger_status_one_it = []
            channel = hcr.add_random_phase(station, sampling_rate)

            # The bandpass for the envelope trigger is included in the trigger module,
            # in the high low the filter is applied externally
            if trigger_name == 'high_low':
                channelBandPassFilter.run(event, station, det, passband=passband_trigger,
                                      filter_type='butter', order=order_trigger)

                triggerSimulator.run(event, station, det, threshold_high=threshold, threshold_low=-threshold,
                                     coinc_window = coinc_window, number_concidences = number_coincidences,
                                     triggered_channels = triggered_channels, trigger_name = trigger_name)

            if trigger_name == 'envelope':
                triggerSimulator.run(event, station, det, passband_trigger, order_trigger, threshold, coinc_window,
                number_coincidences=number_coincidences, triggered_channels=triggered_channels, trigger_name=trigger_name)

            # trigger status for one iteration in the loop
            trigger_status_one_it = station.get_trigger(trigger_name).has_triggered()
            # trigger status for all iteration
            trigger_status_per_all_it.append(trigger_status_one_it)

        # here it is checked, how many of the triggers in n_iteration are triggered true.
        # If it is more than 1, the threshold is increased with n_thres.
        if np.sum(trigger_status_per_all_it) > number_of_allowed_trigger:
            number_of_trigger = np.sum(trigger_status_per_all_it)
            trigger_efficiency_per_tt = np.sum(trigger_status_per_all_it) / len(trigger_status_per_all_it)
            trigger_rate_per_tt = (1 / channel_trace_time_interval) * trigger_efficiency_per_tt

            trigger_rate.append(trigger_rate_per_tt)
            trigger_efficiency.append(trigger_efficiency_per_tt)
            continue;

        elif n_it == (n_iterations-1):
            number_of_trigger = np.sum(trigger_status_per_all_it)
            trigger_efficiency_per_tt = np.sum(trigger_status_per_all_it) / len(trigger_status_per_all_it)
            trigger_rate_per_tt = (1 / channel_trace_time_interval) * trigger_efficiency_per_tt

            trigger_rate.append(trigger_rate_per_tt)
            trigger_efficiency.append(trigger_efficiency_per_tt)

            thresholds = np.array(thresholds)
            trigger_rate = np.array(trigger_rate)
            trigger_efficiency = np.array(trigger_efficiency)

            dic = {}
            dic['T_noise'] = Tnoise
            dic['Vrms_thermal_noise'] = Vrms_thermal_noise
            dic['thresholds'] = thresholds
            dic['efficiency'] = trigger_efficiency
            dic['trigger_rate'] = trigger_rate
            dic['n_iterations'] = n_iterations * n_random_phase  # from phase change in galactic noise
            dic['passband_trigger'] = passband_trigger
            dic['coinc_window'] = coinc_window
            dic['order_trigger'] = order_trigger
            dic['number_coincidences'] = number_coincidences
            dic['detector_file'] = detector_file
            dic['triggered_channels'] = triggered_channels
            dic['default_station'] = default_station
            dic['sampling_rate'] = sampling_rate
            dic['T_noise_min_freq'] = T_noise_min_freq
            dic['T_noise_max_freq '] = T_noise_max_freq
            dic['galactic_noise_n_side'] = galactic_noise_n_side
            dic['galactic_noise_interpolation_frequencies_start'] = galactic_noise_interpolation_frequencies_start
            dic['galactic_noise_interpolation_frequencies_stop'] = galactic_noise_interpolation_frequencies_stop
            dic['galactic_noise_interpolation_frequencies_step'] = galactic_noise_interpolation_frequencies_step
            dic['station_time'] = station_time
            dic['station_time_random'] = station_time_random
            dic['hardware_response'] = hardware_response
            dic['trigger_name'] = trigger_name
            dic['n_random_phase'] = n_random_phase

            if os.path.isdir(os.path.join(abs_output_path, 'output_threshold_estimate')) == False:
                os.mkdir(os.path.join(abs_output_path, 'output_threshold_estimate'))

            output_file = 'output_threshold_estimate/estimate_threshold_{}_pb_{:.0f}_{:.0f}_i{}.json'.format(
                trigger_name, passband_trigger[0]/units.MHz,passband_trigger[1]/units.MHz, len(trigger_status_per_all_it))

            abs_path_output_file = os.path.normpath(os.path.join(abs_output_path, output_file))

            with open(abs_path_output_file, 'w') as outfile:
                json.dump(dic, outfile, cls=hcr.NumpyEncoder, indent=4)


