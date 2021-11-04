import numpy as np
import os, glob
import json, pickle
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

logger = logging.getLogger()
logger.setLevel(logging.INFO)

'''
This script reconstructs the air shower (stored in air_shower_sim as hdf5 files) with 
the trigger parameters calculated before'''

parser = argparse.ArgumentParser(description='Run air shower Reconstruction')

parser.add_argument('detector_file', type=str, nargs='?',
                    default='../example_data/arianna_station_32.json',
                    help='choose detector for air shower simulation')
parser.add_argument('default_station', type=int, nargs='?',
                    default=32, help='define default station for detector')
parser.add_argument('config_file', type=str, nargs='?',
                    default='config/final_config', help='settings from the ntr results')
parser.add_argument('eventlist', type=list, nargs='?',
                    default=['../example_data/example_event.h5'], help='list with event files')
parser.add_argument('number', type=int, nargs='?',
                    default=0, help='number of element in eventlist')
parser.add_argument('output_filename', type=str, nargs='?',
                    default='output_air_shower_reco/air_shower_reco_', help='begin of output filename')

args = parser.parse_args()

if '.p' in args.eventlist:
    eventlist = pickle.load(open(args.eventlist, 'br'))
else:
    eventlist = args.eventlist

os.makedirs(args.output_filename, exist_ok=True)

config_file = glob.glob('{}*'.format(args.config_file))[0]
with open(config_file, 'r') as fp:
    cfg = json.load(fp)

input_files = eventlist[args.number]

logger.info(f"Apply {config_file} on {args.detector_file} with default station {args.default_station}")


det = GenericDetector(json_filename=args.detector_file, default_station=args.default_station)
det.update(datetime.datetime(2019, 10, 1))

station_ids = det.get_station_ids()
station_id = station_ids[0]
channel_ids = det.get_channel_ids(station_id)

# initialize all modules that are needed for processing
# provide input parameters that are to remain constant during processing
readCoREASStation = NuRadioReco.modules.io.coreas.readCoREASStation.readCoREASStation()
readCoREASStation.begin([input_files], args.default_station, debug=False)
simulationSelector = NuRadioReco.modules.io.coreas.simulationSelector.simulationSelector()
simulationSelector.begin()
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin(debug=False)
hardwareResponseIncorporator = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
channelGenericNoiseAdder.begin()
channelGalacticNoiseAdder = NuRadioReco.modules.channelGalacticNoiseAdder.channelGalacticNoiseAdder()
channelGalacticNoiseAdder.begin(
    n_side=cfg['galactic_noise_n_side'],
    interpolation_frequencies=np.arange(cfg['galactic_noise_interpolation_frequencies_start'],
                                        cfg['galactic_noise_interpolation_frequencies_stop'],
                                        cfg['galactic_noise_interpolation_frequencies_step']))
if cfg['trigger_name'] == 'high_low':
    triggerSimulator = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
    triggerSimulator.begin()

if cfg['trigger_name'] == 'envelope':
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
eventWriter.begin(args.output_filename + 'pb_' + str(int(cfg['passband_trigger'][0] / units.MHz))
                  + '_' + str(int(cfg['passband_trigger'][1] / units.MHz))
                  + '_tt_' + str(round(cfg['final_threshold'], 6)) + '_' + str(args.number) + '.nur')

# Loop over all events in file as initialized in readCoRREAS and perform analysis
i = 0
for evt in readCoREASStation.run(det):
    for sta in evt.get_stations():
        if i == 10:  # use this if you want to test something or if you want only 10 position
            break
        logger.info("processing event {:d} with id {:d}".format(i, evt.get_id()))

        station = evt.get_station(args.default_station)
        if cfg['station_time_random']:
            station = hcr.set_random_station_time(station, cfg['station_time'])

        efieldToVoltageConverter.run(evt, sta, det)
        eventTypeIdentifier.run(evt, sta, "forced", 'cosmic_ray')
        channelGenericNoiseAdder.run(evt, sta, det, amplitude=cfg['Vrms_thermal_noise'],
                                     min_freq=cfg['T_noise_min_freq'], max_freq=cfg['T_noise_max_freq'],
                                     type='rayleigh')
        channelGalacticNoiseAdder.run(evt, sta, det)

        if cfg['hardware_response']:
            hardwareResponseIncorporator.run(evt, sta, det, sim_to_data=True)

        # The bandpass for the envelope trigger is included in the trigger module,
        # in the high low the filter is applied externally
        if cfg['trigger_name'] == 'high_low':
            channelBandPassFilter.run(evt, sta, det, passband=cfg['passband_trigger'],
                                      filter_type='butter', order=cfg['order_trigger'])

            triggerSimulator.run(evt, sta, det,
                                 threshold_high=cfg['final_threshold'],
                                 threshold_low=-cfg['final_threshold'],
                                 coinc_window=cfg['coinc_window'],
                                 number_concidences=cfg['number_coincidences'],
                                 triggered_channels=cfg['triggered_channels'],
                                 trigger_name='{}_pb_{:.0f}_{:.0f}_tt_{:.2f}'.format(
                                     cfg['trigger_name'],
                                     cfg['passband_trigger'][0] / units.MHz,
                                     cfg['passband_trigger'][1] / units.MHz,
                                     cfg['final_threshold'] / units.mV))

        if cfg['trigger_name'] == 'envelope':
            triggerSimulator.run(evt, sta, det,
                                 passband=cfg['passband_trigger'],
                                 order=cfg['order_trigger'],
                                 number_coincidences=cfg['number_coincidences'],
                                 threshold=cfg['final_threshold'],
                                 coinc_window=cfg['coinc_window'],
                                 triggered_channels=cfg['triggered_channels'],
                                 trigger_name='{}_pb_{:.0f}_{:.0f}_tt_{:.2f}'.format(
                                     cfg['trigger_name'],
                                     cfg['passband_trigger'][0] / units.MHz,
                                     cfg['passband_trigger'][1] / units.MHz,
                                     cfg['final_threshold'] / units.mV))

        channelResampler.run(evt, sta, det, sampling_rate=cfg['sampling_rate'])

        electricFieldResampler.run(evt, sta, det, sampling_rate=cfg['sampling_rate'])
        i += 1
    eventWriter.run(evt, mode='micro')  # here you can change what should be stored in the nur files
nevents = eventWriter.end()
