import numpy as np
import os, scipy, sys
import copy
import datetime
import glob
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
import NuRadioReco.modules.cosmicRayIdentifier
import NuRadioReco.modules.channelStopFilter
import NuRadioReco.modules.channelSignalReconstructor
import NuRadioReco.modules.correlationDirectionFitter
import NuRadioReco.modules.voltageToEfieldConverter
import NuRadioReco.modules.electricFieldSignalReconstructor
import NuRadioReco.modules.voltageToAnalyticEfieldConverter
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.electricFieldResampler

import NuRadioReco.modules.io.eventWriter

from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import stationParameters as stnp

# Logging level
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('FullExample')

plt.switch_backend('agg')


"""
Here, we show an example reconstruction of CoREAS data. A variety of modules
are being used. Please refer to details in the modules themselves.

Input parameters (all with a default provided)
---------------------

Command line input:
    python FullReconstruction.py station_id input_file detector_file

station_id: int
            station id to be used, default 32
input_file: str
            CoREAS simulation file, default example data
detector_file: str
            path to json detector database, default given
"""

try:
    station_id = int(sys.argv[1])  # specify station id
    input_file = sys.argv[2] # file with coreas simulations
except:
    print("Usage: python FullReconstruction.py station_id input_file detector")
    station_id = 32
    input_file = "example_data/example_event.h5"
    print("Using default station {}".format(32))

if(station_id == 32):
    triggered_channels = [0,1,2,3]
    used_channels_efield = [0, 1, 2, 3]
    used_channels_fit = [0, 1, 2, 3]
    channel_pairs = ((0, 2), (1, 3))
else:
    print("Default channels not defined for station_id != 32")

try:
    detector_file = sys.argv[3]
    print("Using {0} as detector".format(detector_file))
except:
    print("Using default file for detector")
    detector_file = '../examples/example_data/arianna_detector_db.json'


det = detector.Detector(json_filename=detector_file) # detector file
det.update(datetime.datetime(2018, 10, 1))

dir_path = os.path.dirname(os.path.realpath(__file__)) # get the directory of this file

# initialize all modules that are needed for processing
# provide input parameters that are to remain constant during processung
readCoREAS = NuRadioReco.modules.io.coreas.readCoREAS.readCoREAS()
readCoREAS.begin([input_file], station_id, n_cores=10, max_distance=None)
simulationSelector = NuRadioReco.modules.io.coreas.simulationSelector.simulationSelector()
simulationSelector.begin()
efieldToVoltageConverter =  NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin(debug=False)
hardwareResponseIncorporator = NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator.hardwareResponseIncorporator()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
channelGenericNoiseAdder.begin()
triggerSimulator = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()
triggerSimulator.begin()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelBandPassFilter.begin()
cosmicRayIdentifier = NuRadioReco.modules.cosmicRayIdentifier.cosmicRayIdentifier()
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
output_filename = "MC_example_station_{}.nur".format(station_id)
eventWriter.begin(output_filename)

# Loop over all events in file as initialized in readCoRREAS and perform analysis
for iE, evt in enumerate(readCoREAS.run(detector=det)):
    logger.info("processing event {:d} with id {:d}".format(iE, evt.get_id()))
    station = evt.get_station(station_id)

    if simulationSelector.run(evt, station.get_sim_station(), det):

        efieldToVoltageConverter.run(evt, station, det)

        hardwareResponseIncorporator.run(evt, station, det, sim_to_data=True)

        channelGenericNoiseAdder.run(evt, station, det, type = "rayleigh", amplitude = 20* units.mV)

        triggerSimulator.run(evt, station,det, number_concidences = 2, threshold = 100 *units.mV)

        if station.get_trigger('default_simple_threshold').has_triggered():

            channelBandPassFilter.run(evt, station, det, passband=[80 * units.MHz, 500 * units.MHz], filter_type='butter', order = 10)

            cosmicRayIdentifier.run(evt, station, "forced")

            channelStopFilter.run(evt, station, det)

            channelBandPassFilter.run(evt, station, det, passband=[60 * units.MHz, 600 * units.MHz], filter_type='rectangular')

            channelSignalReconstructor.run(evt, station, det)

            hardwareResponseIncorporator.run(evt, station, det)

            correlationDirectionFitter.run(evt, station, det, n_index=1., channel_pairs=channel_pairs)

            voltageToEfieldConverter.run(evt, station, det, use_channels=used_channels_efield)

            electricFieldSignalReconstructor.run(evt, station, det)
            voltageToAnalyticEfieldConverter.run(evt, station, det, use_channels=used_channels_efield, bandpass=[80*units.MHz, 500*units.MHz], useMCdirection=False)

            channelResampler.run(evt, station, det, sampling_rate=1 * units.GHz)
            
            electricFieldResampler.run(evt, station, det, sampling_rate=1 * units.GHz)

            eventWriter.run(evt)


nevents = eventWriter.end()
print("Finished processing, {} events".format(nevents))








