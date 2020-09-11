#!/usr/bin/env python
import os
import sys
import datetime
import matplotlib
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units
from NuRadioReco.detector import detector
import NuRadioReco.modules.io.coreas.readCoREAS
import NuRadioReco.modules.io.coreas.simulationSelector
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.eventTypeIdentifier
import NuRadioReco.modules.channelStopFilter
import NuRadioReco.modules.channelSignalReconstructor
import NuRadioReco.modules.correlationDirectionFitter
import NuRadioReco.modules.voltageToEfieldConverter
import NuRadioReco.modules.electricFieldSignalReconstructor
import NuRadioReco.modules.electricFieldBandPassFilter
import NuRadioReco.modules.voltageToAnalyticEfieldConverter
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.electricFieldResampler
import NuRadioReco.modules.io.eventWriter

# Logging level
import logging
from NuRadioReco.modules.base import module
logger = module.setup_logger(name='NuRadioReco', level=logging.WARNING)

matplotlib.use('agg')
plt.switch_backend('agg')


"""
Here, we show an example reconstruction of CoREAS data. A variety of modules
are being used. Please refer to details in the modules themselves.

Input parameters (all with a default provided)
---------------------

Command line input:
    python FullReconstruction.py station_id input_file detector_file templates

station_id: int
            station id to be used, default 32
input_file: str
            CoREAS simulation file, default example data
detector_file: str
            path to json detector database, default given
template_path: str
            path to signal templates, default given

"""

dir_path = os.path.dirname(os.path.realpath(__file__))  # get the directory of this file

try:
    station_id = int(sys.argv[1])  # specify station id
    input_file = sys.argv[2]    # file with coreas simulations
except:
    logger.warning("Usage: python FullReconstruction.py station_id input_file detector templates")
    station_id = 32
    input_file = os.path.join(dir_path, "../../examples/example_data/example_event.h5")
    logger.warning("Using default station {}".format(32))

if(station_id == 32):
    triggered_channels = [0, 1, 2, 3]
    used_channels_efield = [0, 1, 2, 3]
    used_channels_fit = [0, 1, 2, 3]
    channel_pairs = ((0, 2), (1, 3))
else:
    logger.warning("Default channels not defined for station_id != 32")

try:
    detector_file = sys.argv[3]
    logger.info("Using {0} as detector ".format(detector_file))
except:

    logger.warning("Using default file for detector")
    detector_file = os.path.join(dir_path, "../../examples/example_data/arianna_station_32.json")

det = detector.Detector(json_filename=detector_file)    # detector file
det.update(datetime.datetime(2018, 10, 1))

# initialize all modules that are needed for processing
# provide input parameters that are to remain constant during processung
readCoREAS = NuRadioReco.modules.io.coreas.readCoREAS.readCoREAS()
readCoREAS.begin([input_file], station_id, n_cores=10, max_distance=None, seed=0)
simulationSelector = NuRadioReco.modules.io.coreas.simulationSelector.simulationSelector()
simulationSelector.begin()
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin(debug=False)
hardwareResponseIncorporator = NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator.hardwareResponseIncorporator()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
channelGenericNoiseAdder.begin(seed=1)
triggerSimulator = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()
triggerSimulator.begin()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelBandPassFilter.begin()
eventTypeIdentifier = NuRadioReco.modules.eventTypeIdentifier.eventTypeIdentifier()
channelStopFilter = NuRadioReco.modules.channelStopFilter.channelStopFilter()
channelSignalReconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
channelSignalReconstructor.begin(signal_window_start=20 * units.ns, signal_window_length=80 * units.ns, noise_window_start=150 * units.ns, noise_window_length=200 * units.ns)
correlationDirectionFitter = NuRadioReco.modules.correlationDirectionFitter.correlationDirectionFitter()
voltageToEfieldConverter = NuRadioReco.modules.voltageToEfieldConverter.voltageToEfieldConverter()

electricFieldSignalReconstructor = NuRadioReco.modules.electricFieldSignalReconstructor.electricFieldSignalReconstructor()
electricFieldSignalReconstructor.begin()

voltageToAnalyticEfieldConverter = NuRadioReco.modules.voltageToAnalyticEfieldConverter.voltageToAnalyticEfieldConverter()
voltageToAnalyticEfieldConverter.begin()

electricFieldResampler = NuRadioReco.modules.electricFieldResampler.electricFieldResampler()
electricFieldResampler.begin()
electricFieldBandPassFilter = NuRadioReco.modules.electricFieldBandPassFilter.electricFieldBandPassFilter()

channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelResampler.begin()
eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
output_filename = "MC_example_station_{}.nur".format(station_id)
eventWriter.begin(output_filename)


event_counter = 0
# Loop over all events in file as initialized in readCoRREAS and perform analysis
for iE, evt in enumerate(readCoREAS.run(detector=det)):
    logger.warning("Processing event number {}".format(event_counter))
    logger.info("processing event {:d} with id {:d}".format(iE, evt.get_id()))
    station = evt.get_station(station_id)

    if simulationSelector.run(evt, station.get_sim_station(), det):

        efieldToVoltageConverter.run(evt, station, det)

        hardwareResponseIncorporator.run(evt, station, det, sim_to_data=True)

        channelGenericNoiseAdder.run(evt, station, det, type="rayleigh", amplitude=20 * units.mV)

        triggerSimulator.run(evt, station, det, number_concidences=2, threshold=100 * units.mV)

        if station.get_trigger('default_simple_threshold').has_triggered():

            channelBandPassFilter.run(evt, station, det, passband=[80 * units.MHz, 500 * units.MHz], filter_type='butter', order=10)

            eventTypeIdentifier.run(evt, station, "forced", 'cosmic_ray')

            channelStopFilter.run(evt, station, det)

            channelBandPassFilter.run(evt, station, det, passband=[60 * units.MHz, 600 * units.MHz], filter_type='rectangular')
            channelSignalReconstructor.run(evt, station, det)

            hardwareResponseIncorporator.run(evt, station, det)

            correlationDirectionFitter.run(evt, station, det, n_index=1., channel_pairs=channel_pairs)

            voltageToEfieldConverter.run(evt, station, det, use_channels=used_channels_efield)

            electricFieldBandPassFilter.run(evt, station, det, passband=[80 * units.MHz, 300 * units.MHz])

            electricFieldSignalReconstructor.run(evt, station, det)

            voltageToAnalyticEfieldConverter.run(evt, station, det, use_channels=used_channels_efield, bandpass=[80 * units.MHz, 500 * units.MHz], use_MC_direction=False)

            channelResampler.run(evt, station, det, sampling_rate=1 * units.GHz)

            electricFieldResampler.run(evt, station, det, sampling_rate=1 * units.GHz)

            eventWriter.run(evt)

    event_counter += 1
    if event_counter > 2:
        break
nevents = eventWriter.end()
logger.warning("Finished processing, {} events".format(event_counter))
