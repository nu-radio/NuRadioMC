import numpy as np
import os
import json
import helper_cr_eff as hcr
import datetime
from NuRadioReco.utilities import units
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
import NuRadioReco.modules.io.eventWriter
import logging
import argparse
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('air_shower_reco')

'''
This script reconstructs the air shower (stored in air_shower_sim as hdf5 files) with 
the trigger parameters calculated in step 1-3'''

parser = argparse.ArgumentParser(description='Run air shower Reconstruction')

parser.add_argument('result_dict', type=str, nargs='?',
                    default='results/ntr/dict_ntr_high_low_pb_80_180.json', help='settings from the ntr results')
parser.add_argument('eventlist', type=list, nargs='?',
                    default=['../example_data/example_event.h5'], help='list with event files')
parser.add_argument('number', type=int, nargs='?',
                    default=0, help= 'number of element in eventlist')
parser.add_argument('output_filename', type=str, nargs='?',
                    default='output_air_shower_reco/air_shower_reco_', help='begin of output filename')

args = parser.parse_args()
result_dict = args.result_dict
eventlist = args.eventlist
number = args.number
output_filename = args.output_filename

#eventlist = pickle.load(open(cfg['eventlist'], 'br'))

os.makedirs(output_filename, exist_ok=True)

with open(result_dict, 'r') as fp:
    data = json.load(fp)

detector_file = data['detector_file']
default_station = data['default_station']
sampling_rate = data['sampling_rate']
station_time = data['station_time']
station_time_random = data['station_time_random']
triggered_channels = data['triggered_channels']

Vrms_thermal_noise = data['Vrms_thermal_noise']
T_noise = data['T_noise']
T_noise_min_freq = data['T_noise_min_freq']
T_noise_max_freq = data['T_noise_max_freq ']

galactic_noise_n_side = data['galactic_noise_n_side']
galactic_noise_interpolation_frequencies_start = data['galactic_noise_interpolation_frequencies_start']
galactic_noise_interpolation_frequencies_stop = data['galactic_noise_interpolation_frequencies_stop']
galactic_noise_interpolation_frequencies_step = data['galactic_noise_interpolation_frequencies_step']

trigger_name = data['trigger_name']
passband_trigger = data['passband_trigger']
number_coincidences = data['number_coincidences']
coinc_window = data['coinc_window']
order_trigger = data['order_trigger']
trigger_thresholds = data['threshold']
n_iterations = data['iteration']
hardware_response = data['hardware_response']

trigger_rate = np.array(data['trigger_rate'])
threshold_tested = data['threshold']
zeros = np.where(trigger_rate == 0)[0]
first_zero = zeros[0]  # gives the index of the element where the trigger rate is zero for the first time
trigger_threshold = threshold_tested[first_zero]

input_files = eventlist[number]

used_channels_efield = triggered_channels
used_channels_fit = triggered_channels

det = GenericDetector(json_filename=detector_file, default_station=default_station) # detector file
det.update(datetime.datetime(2019, 10, 1))

station_ids = det.get_station_ids()
station_id = station_ids[0]
channel_ids = det.get_channel_ids(station_id)

# initialize all modules that are needed for processing
# provide input parameters that are to remain constant during processing
readCoREASStation = NuRadioReco.modules.io.coreas.readCoREASStation.readCoREASStation()
readCoREASStation.begin([input_files], default_station, debug=False)
simulationSelector = NuRadioReco.modules.io.coreas.simulationSelector.simulationSelector()
simulationSelector.begin()
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin(debug=False)
hardwareResponseIncorporator = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
channelGenericNoiseAdder.begin()
channelGalacticNoiseAdder = NuRadioReco.modules.channelGalacticNoiseAdder.channelGalacticNoiseAdder()
channelGalacticNoiseAdder.begin(n_side=galactic_noise_n_side,
                                interpolation_frequencies=
                                np.arange(galactic_noise_interpolation_frequencies_start,
                                          galactic_noise_interpolation_frequencies_stop,
                                          galactic_noise_interpolation_frequencies_step))
if trigger_name == 'high_low':
    triggerSimulator = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
    triggerSimulator.begin()

if trigger_name == 'envelope':
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
eventWriter.begin(output_filename + 'pb_' + str(int(passband_trigger[0]/units.MHz))
                  + '_' + str(int(passband_trigger[1]/units.MHz))
                  + '_tt_' + str(round(trigger_threshold, 6)) + '_' + str(number) + '.nur')

# Loop over all events in file as initialized in readCoRREAS and perform analysis
i = 0
for evt in readCoREASStation.run(det):
    for sta in evt.get_stations():
        if i == 10: # use this if you want to test something or if you want only 30 position
           break
        logger.info("processing event {:d} with id {:d}".format(i, evt.get_id()))

        station = evt.get_station(default_station)
        if station_time_random == True:
            station = hcr.set_random_station_time(station)

        efieldToVoltageConverter.run(evt, sta, det)
        eventTypeIdentifier.run(evt, sta, "forced", 'cosmic_ray')
        channelGenericNoiseAdder.run(evt, sta, det, amplitude=Vrms_thermal_noise, min_freq=T_noise_min_freq,
                                 max_freq=T_noise_max_freq, type='rayleigh', bandwidth=None)

        channelGalacticNoiseAdder.run(evt, sta, det)

        if hardware_response == True:
            hardwareResponseIncorporator.run(evt, sta, det, sim_to_data=True)

        # The bandpass for the envelope trigger is included in the trigger module,
        # in the high low the filter is applied externally
        if trigger_name == 'high_low':
            channelBandPassFilter.run(evt, sta, det, passband=passband_trigger,
                                      filter_type='butter', order=order_trigger)

            triggerSimulator.run(evt, sta, det, threshold_high=trigger_threshold, threshold_low=-trigger_threshold,
                                 coinc_window=coinc_window, number_concidences=number_coincidences,
                                 triggered_channels=triggered_channels, trigger_name='{}_pb_{:.0f}_{:.0f}_tt_{:.2f}'.format(trigger_name ,passband_trigger[0]/units.MHz, passband_trigger[1]/units.MHz, trigger_threshold/units.mV))

        if trigger_name == 'envelope':
            triggerSimulator.run(evt, sta, det, passband=passband_trigger, order=order_trigger,
                             number_coincidences=number_coincidences, threshold=trigger_threshold,
                             coinc_window=coinc_window, triggered_channels=triggered_channels,
                trigger_name='{}_pb_{:.0f}_{:.0f}_tt_{:.2f}'.format(trigger_name, passband_trigger[0]/units.MHz, passband_trigger[1]/units.MHz, trigger_threshold/units.mV))

        channelResampler.run(evt, sta, det, sampling_rate=sampling_rate)

        electricFieldResampler.run(evt, sta, det, sampling_rate=sampling_rate)
        i += 1
    eventWriter.run(evt, mode='micro')  # here you can change what should be stored in the nur files
nevents = eventWriter.end()
