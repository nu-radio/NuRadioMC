import os
import sys
import datetime
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units
from NuRadioReco.detector import detector
import NuRadioReco.modules.io.coreas.readCoREAS
import NuRadioReco.modules.io.coreas.simulationSelector
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.electricFieldBandPassFilter
import NuRadioReco.modules.eventTypeIdentifier
import NuRadioReco.modules.channelStopFilter
import NuRadioReco.modules.voltageToEfieldConverter
import NuRadioReco.modules.electricFieldSignalReconstructor
import NuRadioReco.modules.voltageToAnalyticEfieldConverter
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.io.eventWriter
from NuRadioReco.modules.base import module

# Logging level
import logging
logger = module.setup_logger(level=logging.INFO)

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
    input_file = sys.argv[2]  # file with coreas simulations
except:
    print("Usage: python SimpleMCReconstruction.py station_id input_file detector")
    station_id = 32
    input_file = "example_data/example_event.h5"
    print("Using default station {}".format(32))

if(station_id == 32):
    triggered_channels = [0, 1, 2, 3]
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
    detector_file = 'example_data/arianna_station_32.json'

det = detector.Detector(json_filename=detector_file)  # detector file
det.update(datetime.datetime(2018, 10, 1))

dir_path = os.path.dirname(os.path.realpath(__file__))  # get the directory of this file

# initialize all modules that are needed for processing
# provide input parameters that are to remain constant during processung
readCoREAS = NuRadioReco.modules.io.coreas.readCoREAS.readCoREAS()
readCoREAS.begin([input_file], station_id, n_cores=10, max_distance=None)

simulationSelector = NuRadioReco.modules.io.coreas.simulationSelector.simulationSelector()
simulationSelector.begin()

electricFieldBandPassFilter = NuRadioReco.modules.electricFieldBandPassFilter.electricFieldBandPassFilter()

efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin(debug=False)

channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
channelGenericNoiseAdder.begin()

channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelBandPassFilter.begin()

channelStopFilter = NuRadioReco.modules.channelStopFilter.channelStopFilter()

voltageToEfieldConverter = NuRadioReco.modules.voltageToEfieldConverter.voltageToEfieldConverter()

electricFieldSignalReconstructor = \
    NuRadioReco.modules.electricFieldSignalReconstructor.electricFieldSignalReconstructor()
electricFieldSignalReconstructor.begin()

voltageToAnalyticEfieldConverter = \
    NuRadioReco.modules.voltageToAnalyticEfieldConverter.voltageToAnalyticEfieldConverter()
voltageToAnalyticEfieldConverter.begin()

eventTypeIdentifier = NuRadioReco.modules.eventTypeIdentifier.eventTypeIdentifier()

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

        eventTypeIdentifier.run(evt, station, 'forced', 'cosmic_ray')

        efieldToVoltageConverter.run(evt, station, det)

        channelGenericNoiseAdder.run(evt, station, det, min_freq=25 * units.MHz, type="rayleigh", amplitude=1 * units.mV)

        channelStopFilter.run(evt, station, det)
        channelResampler.run(evt, station, det, sampling_rate=0.8 * units.GHz)

        # voltageToAnalyticEfieldConverter expect butter filter
        channelBandPassFilter.run(evt, station, det, passband=[20 * units.MHz, 90 * units.MHz], filter_type='rectangular')
        channelBandPassFilter.run(evt, station, det, passband=[30 * units.MHz, 80 * units.MHz], filter_type='butter', order=10)

        # traditional
        voltageToEfieldConverter.run(evt, station, det, use_channels=used_channels_efield, use_MC_direction=True)
        electricFieldBandPassFilter.run(evt, station, det, passband=[30 * units.MHz, 80 * units.MHz], filter_type='butter', order=10)
        electricFieldSignalReconstructor.run(evt, station, det)

        # channelSignalReconstructor.run(evt, station, det)
        # new analytic approach
        voltageToAnalyticEfieldConverter.run(evt, station, det, use_channels=used_channels_efield, bandpass=[30 * units.MHz, 80 * units.MHz], use_MC_direction=True)

        eventWriter.run(evt)

nevents = eventWriter.end()
print("Finished processing, {} events".format(nevents))
