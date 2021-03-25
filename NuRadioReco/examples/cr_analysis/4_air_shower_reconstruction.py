import numpy as np
import os, scipy, sys
import yaml
import scipy.constants
import datetime
import matplotlib.pyplot as plt
import pickle
import pygdsm
import astropy
from NuRadioReco.utilities import units, io_utilities
from NuRadioReco.detector.generic_detector import GenericDetector
import NuRadioReco.modules.io.coreas.readCoREASStation
import NuRadioReco.modules.io.coreas.simulationSelector
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.channelGalacticNoiseAdder
import NuRadioReco.modules.trigger.envelopeTrigger
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.eventTypeIdentifier
import NuRadioReco.modules.channelStopFilter
import NuRadioReco.modules.channelSignalReconstructor
import NuRadioReco.modules.correlationDirectionFitter
import NuRadioReco.modules.voltageToEfieldConverter
import NuRadioReco.modules.electricFieldSignalReconstructor
import NuRadioReco.modules.voltageToAnalyticEfieldConverter
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.electricFieldResampler
import NuRadioReco.framework.event
import NuRadioReco.modules.io.eventWriter
import logging
import argparse
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('FullExample')

'''
This script reconstructs the air shower with the parameters calculated in step 1-3.
'''

parser = argparse.ArgumentParser(description='Run FullReconstruction')
parser.add_argument('config_file', type=str, nargs='?', default = 'config_file_air_shower_reco.yml', help = 'config file with eventlist')
parser.add_argument('result_dict', type=str, nargs='?', default = 'results/ntr/dict_ntr_pb_80_180.pickle', help = 'settings from the results from threshold analysis')
parser.add_argument('number', type=int, nargs='?', default = 0, help = 'number of element in eventlist')

args = parser.parse_args()
config_file = args.config_file
result_dict = args.result_dict
number = args.number

with open(config_file, 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)
# print('config file', cfg)

eventlist = cfg['eventlist']
output_filename = cfg['output_filename']

data = io_utilities.read_pickle(result_dict, encoding='latin1')
# print('data', data)

detector_file = data['detector_file']
default_station = data['default_station']
sampling_rate = data['sampling_rate'] * units.gigahertz
station_time = data['station_time']
station_time_random = data['station_time_random']

Vrms_thermal_noise = data['Vrms_thermal_noise'] * units.volt
T_noise = data['T_noise'] * units.kelvin
T_noise_min_freq = data['T_noise_min_freq'] * units.megahertz
T_noise_max_freq = data['T_noise_max_freq '] * units.megahertz

galactic_noise_n_side = data['galactic_noise_n_side']
galactic_noise_interpolation_frequencies_step = data['galactic_noise_interpolation_frequencies_step']

passband_trigger = data['passband_trigger']
number_coincidences = data['number_coincidences']
coinc_window = data['coinc_window'] * units.ns
order_trigger = data['order_trigger']
trigger_thresholds = data['threshold']
n_iterations = data['iteration']
hardware_response = data['hardware_response']

trigger_rate = data['trigger_rate']
threshold_tested = data['threshold']
print(trigger_rate)
zeros = np.where(trigger_rate == 0)[0]
#print(zeros)
first_zero = zeros[0]
#print(first_zero)
trigger_threshold = threshold_tested[first_zero] * units.volt
print('threshold', trigger_threshold/units.mV)

input_files = eventlist[number]
print('Input file', input_files)

if(default_station == 101):
    triggered_channels = [16, 19, 22]
    used_channels_efield = [16, 19, 22]
    used_channels_fit = [16, 19, 22]
    channel_pairs = ((16, 19), (16, 22), (19, 22))
else:
    print("Default channels not defined for station_id != 101")

print("Using {0} as detector".format(detector_file))

det = GenericDetector(json_filename=detector_file, default_station=default_station) # detector file
det.update(datetime.datetime(2019, 10, 1))

station_ids = det.get_station_ids()
station_id = station_ids[0]
channel_ids = det.get_channel_ids(station_id)

dir_path = os.path.dirname(os.path.realpath(__file__)) # get the directory of this file

# initialize all modules that are needed for processing
# provide input parameters that are to remain constant during processung
readCoREASStation = NuRadioReco.modules.io.coreas.readCoREASStation.readCoREASStation()
readCoREASStation.begin([input_files], default_station, debug=False)
simulationSelector = NuRadioReco.modules.io.coreas.simulationSelector.simulationSelector()
simulationSelector.begin()
efieldToVoltageConverter =  NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin(debug=False)
hardwareResponseIncorporator = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
channelGenericNoiseAdder.begin()
channelGalacticNoiseAdder = NuRadioReco.modules.channelGalacticNoiseAdder.channelGalacticNoiseAdder()
channelGalacticNoiseAdder.begin(n_side=4, interpolation_frequencies=np.arange(10, 1100, galactic_noise_interpolation_frequencies_step) * units.MHz)
triggerSimulator = NuRadioReco.modules.trigger.envelopeTrigger.triggerSimulator()
triggerSimulator.begin()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelBandPassFilter.begin()
eventTypeIdentifier = NuRadioReco.modules.eventTypeIdentifier.eventTypeIdentifier()
channelStopFilter = NuRadioReco.modules.channelStopFilter.channelStopFilter()
channelSignalReconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
channelSignalReconstructor.begin()
correlationDirectionFitter = NuRadioReco.modules.correlationDirectionFitter.correlationDirectionFitter()
voltageToEfieldConverter = NuRadioReco.modules.voltageToEfieldConverter.voltageToEfieldConverter()
electricFieldSignalReconstructor = NuRadioReco.modules.electricFieldSignalReconstructor.electricFieldSignalReconstructor()
electricFieldSignalReconstructor.begin()
voltageToAnalyticEfieldConverter = NuRadioReco.modules.voltageToAnalyticEfieldConverter.voltageToAnalyticEfieldConverter()
voltageToAnalyticEfieldConverter.begin()
electricFieldResampler = NuRadioReco.modules.electricFieldResampler.electricFieldResampler()
electricFieldResampler.begin()
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelResampler.begin()
eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
eventWriter.begin(output_filename + 'pb_' + str(passband_trigger[0]) + str(passband_trigger[1]) + '_tt_' + str(trigger_threshold.round(4)) + '.nur')

i = 0
# Loop over all events in file as initialized in readCoRREAS and perform analysis
for evt in readCoREASStation.run(det):
    for sta in evt.get_stations():
        logger.info("processing event {:d} with id {:d}".format(i, evt.get_id()))

        station = evt.get_station(default_station)
        if station_time_random == True:
            random_generator_hour = np.random.RandomState()
            hour = random_generator_hour.randint(0, 24)
            if hour < 10:
                station.set_station_time(astropy.time.Time('2019-01-01T0{}:00:00'.format(hour)))
            elif hour >= 10:
                station.set_station_time(astropy.time.Time('2019-01-01T{}:00:00'.format(hour)))

        efieldToVoltageConverter.run(evt, sta, det)
        eventTypeIdentifier.run(evt, sta, "forced", 'cosmic_ray')
        channelGenericNoiseAdder.run(evt, sta, det, amplitude=Vrms_thermal_noise, min_freq=T_noise_min_freq,
                                 max_freq=T_noise_max_freq, type='rayleigh', bandwidth=None)

        channelGalacticNoiseAdder.run(evt, sta, det)

        if hardware_response == True:
            hardwareResponseIncorporator.run(evt, sta, det, sim_to_data=True)

        triggerSimulator.run(evt, sta, det, passband=passband_trigger, order=order_trigger,
                             number_coincidences=number_coincidences, threshold=trigger_threshold,
                             coinc_window=coinc_window,
                             trigger_name='envelope_trigger_pb_{:.0f}_{:.0f}_tt_{:.2f}'.format(passband_trigger[0]/units.MHz, passband_trigger[1]/units.MHz, trigger_threshold/units.mV))


        ##channelSignalReconstructor.run(evt, sta, det)

        ##correlationDirectionFitter.run(evt, sta, det, n_index=1., channel_pairs=channel_pairs)

        #voltageToEfieldConverter.run(evt, sta, det, use_channels=used_channels_efield)

        ##electricFieldSignalReconstructor.run(evt, sta, det)

        #voltageToAnalyticEfieldConverter.run(evt, sta, det, use_channels=used_channels_efield, bandpass=[80*units.MHz, 500*units.MHz], useMCdirection=False)

        channelResampler.run(evt, sta, det, sampling_rate=1 * units.GHz)

        electricFieldResampler.run(evt, sta, det, sampling_rate=1 * units.GHz)
        i += 1
        print('end of loop')

    #eventWriter.run(evt, det, mode='micro')

nevents = eventWriter.end()
print("Finished processing, {} events".format(nevents))
