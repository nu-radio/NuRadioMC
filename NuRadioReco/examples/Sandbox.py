import os
import sys
import datetime
import matplotlib.pyplot as plt
import astropy
from NuRadioReco.utilities import units
import numpy as np
from NuRadioReco.detector import detector
from NuRadioReco.modules.base import module
from NuRadioReco.detector.generic_detector import GenericDetector
import NuRadioReco.modules.io.coreas.readCoREASStationGrid
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.eventTypeIdentifier
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.electricFieldResampler
import NuRadioReco.modules.io.eventWriter
# Logging level
import logging
logger = module.setup_logger(name='NuRadioReco', level=logging.WARNING)

plt.switch_backend('agg')

detector_file='/home/henrichs/software/NuRadioMC/NuRadioReco/examples/RNO_G_LPDAs.json'
default_station=11
input_file = "example_data/example_data.hdf5"

det = GenericDetector(json_filename=detector_file, default_station=default_station, antenna_by_depth=False)
det.update(datetime.datetime(2025, 10, 1))

readCoREASStationGrid = NuRadioReco.modules.io.coreas.readCoREASStationGrid.readCoREAS()
readCoREASStationGrid.begin([input_file], xmin=-500, xmax=500, ymin=-500, ymax=100, n_cores=5, seed=0)
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin(debug=False)
hardwareResponseIncorporator = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
channelGenericNoiseAdder.begin()
electricFieldResampler = NuRadioReco.modules.electricFieldResampler.electricFieldResampler()
electricFieldResampler.begin()
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelResampler.begin()
triggerSimulator = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
triggerSimulator.begin()
eventTypeIdentifier = NuRadioReco.modules.eventTypeIdentifier.eventTypeIdentifier()
eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
eventWriter.begin('reco_sandbox.nur')

# Loop over all events in file as initialized in readCoRREAS and perform analysis
i = 0
for evt in readCoREASStationGrid.run(detector=det):
    for sta in evt.get_stations():
        efieldToVoltageConverter.run(evt, sta, det)
        eventTypeIdentifier.run(evt, sta, "forced", 'cosmic_ray')
        logger.info("processing event {:d} with id {:d}".format(i, evt.get_id()))
        print(np.max(np.abs(sta.get_channel(1).get_trace())))
        channelResampler.run(evt, sta, det, sampling_rate=1 * units.GHz)
        electricFieldResampler.run(evt, sta, det, sampling_rate=1 * units.GHz)
    eventWriter.run(evt, det=det)  # here you can change what should be stored in the nur files
nevents = eventWriter.end()
