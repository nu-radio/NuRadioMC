import numpy as np
import os
import scipy
import sys
import copy
import NuRadioReco.modules.io.coreas.readCoREAS
from NuRadioReco.modules import efieldToVoltageConverter as CefieldToVoltageConverter
import NuRadioReco.modules.io.noise.noiseImporter
from NuRadioReco.modules import channelLengthAdjuster
from NuRadioReco.modules.ARIANNA import hardwareResponseIncorporator as ChardwareResponseIncorporator
from NuRadioReco.modules import channelResampler as CchannelResampler
import NuRadioReco.modules.electricFieldResampler
import NuRadioReco.modules.electricFieldBandPassFilter
import NuRadioReco.modules.channelSignalReconstructor
import NuRadioReco.modules.io.eventWriter
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelTemplateCorrelation
import NuRadioReco.modules.channelStopFilter
import NuRadioReco.modules.templateDirectionFitter
import NuRadioReco.modules.correlationDirectionFitter
import NuRadioReco.modules.voltageToAnalyticEfieldConverter
from NuRadioReco.modules.voltageToEfieldConverter import get_array_of_channels
from NuRadioReco.detector import antennapattern
from NuRadioReco.utilities import units, fft
from NuRadioReco.detector import detector
from NuRadioReco.framework.parameters import stationParameters as stnp
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.cosmicRayIdentifier
import datetime
import glob
import matplotlib.pyplot as plt
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.electricFieldSignalReconstructor
import NuRadioReco.modules.voltageToEfieldConverter
from NuRadioReco.framework.parameters import channelParameters as chp

# Logging level
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('FullExample')

station_id = int(sys.argv[1])  # specify station id
if(station_id == 32):
    triggered_channels = [0,1,2,3]
    used_channels_efield = [0, 1, 2, 3]
    used_channels_fit = [0, 1, 2, 3]
    channel_pairs = ((0, 2), (1, 3))

det = detector.Detector(json_filename='ARIANNA/arianna_detector_db.json'.format(station_id))

NOISE_PATH = "/lustre/fs22/group/radio/plaisier/software/simulations/example" #path to measured noise files

input_files = glob.glob(os.path.join('/lustre/fs22/group/radio/sim/cr/000000', "*.hdf5*")) #file with coreas simulations

dir_path = os.path.dirname(os.path.realpath(__file__)) #get the directory of this file
template_directory = os.path.join(dir_path, '../../ARIANNAreco/analysis/templateGeneration') #path to templates

# initialize all modules
readCoREAS = NuRadioReco.modules.io.coreas.readCoREAS.readCoREAS()
readCoREAS.begin(input_files, station_id, n_cores=10, max_distance=None)
electricFieldResampler = NuRadioReco.modules.electricFieldResampler.electricFieldResampler()
electricFieldResampler.begin()
electricFieldBandPassFilter = NuRadioReco.modules.electricFieldBandPassFilter.electricFieldBandPassFilter()
electricFieldBandPassFilter.begin()
efieldToVoltageConverter = CefieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin(debug=False)
hardwareResponseIncorporator = ChardwareResponseIncorporator.hardwareResponseIncorporator()
channelResampler = CchannelResampler.channelResampler()
channelResampler.begin()
noiseImporter = NuRadioReco.modules.io.noise.noiseImporter.noiseImporter()
noiseImporter.begin(NOISE_PATH, station_id=51)
channelSignalReconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
noise_adder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelBandPassFilter.begin()
channelLengthAdjuster = NuRadioReco.modules.channelLengthAdjuster.channelLengthAdjuster()
channelLengthAdjuster.begin()
channelTemplateCorrelation = NuRadioReco.modules.channelTemplateCorrelation.channelTemplateCorrelation(template_directory = template_directory)
voltageToAnalyticEfieldConverter = NuRadioReco.modules.voltageToAnalyticEfieldConverter.voltageToAnalyticEfieldConverter()
voltageToAnalyticEfieldConverter.begin()
channelStopFilter = NuRadioReco.modules.channelStopFilter.channelStopFilter()
templateDirectionFitter = NuRadioReco.modules.templateDirectionFitter.templateDirectionFitter()
correlationDirectionFitter = NuRadioReco.modules.correlationDirectionFitter.correlationDirectionFitter()
cosmicRayIdentifier = NuRadioReco.modules.cosmicRayIdentifier.cosmicRayIdentifier()
cosmicRayIdentifier.begin()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
channelGenericNoiseAdder.begin()
electricFieldSignalReconstructor = NuRadioReco.modules.electricFieldSignalReconstructor.electricFieldSignalReconstructor()
electricFieldSignalReconstructor.begin()
voltageToEfieldConverter = NuRadioReco.modules.voltageToEfieldConverter.voltageToEfieldConverter()
channelSignalReconstructor.begin()
channelTemplateCorrelation.begin()

output_filename = "outputfilename.nur"
eventWriter.begin(output_filename)

for iE, evt in enumerate(readCoREAS.run(det)):
    
    logger.info("processing event {:d} with id {:d}".format(iE, evt.get_id()))
    station = evt.get_station(station_id)
    
    efieldToVoltageConverter.run(evt, station, det)
        
        hardwareResponseIncorporator.run(evt, station, det, sim_to_data=True)
        
        #channelLengthAdjuster.run(evt, station, det)
        
        #noiseImporter.run(evt, station, det)
        
        channelGenericNoiseAdder.run(evt, station, det, type = "perfect_white", amplitude = 20* units.mV)
        
        triggerSimulator.run(evt, station,det)
        
        if station.get_trigger('default_simple_threshold').has_triggered():
            
            channelBandPassFilter.run(evt, station, det, passband=[80 * units.MHz, 500 * units.MHz], filter_type='butter', order = 10)
            
            channelTemplateCorrelation.run(evt, station, det, cosmic_ray=True, channels_to_use=used_channels_fit)
            
            xcorr = station[stnp.cr_xcorrelations]["cr_avg_xcorr_parallel_crchannels"]
            output_mode = 'full'
            if(xcorr < 0.4):
                output_mode = 'micro'
                continue
        
            cosmicRayIdentifier.run(evt, station, "forced")
            
            channelStopFilter.run(evt, station, det)
            
            channelBandPassFilter.run(evt, station, det, passband=[60 * units.MHz, 600 * units.MHz], filter_type='rectangular')
            
            channelSignalReconstructor.run(evt, station, det)
            
            hardwareResponseIncorporator.run(evt, station, det)
            
            templateDirectionFitter.run(evt, station, det, cosmic_ray=True, channels_to_use=used_channels_fit)
            
            correlationDirectionFitter.run(evt, station, det, n_index=1., channel_pairs=channel_pairs)
            
            voltageToEfieldConverter.run(evt, station, det, debug=1, use_channels=used_channels_efield)
            
            electricFieldSignalReconstructor.run(evt, station, det)
            voltageToAnalyticEfieldConverter.run(evt, station, det, use_channels=used_channels_efield, bandpass=[80*units.MHz, 500*units.MHz], useMCdirection=False)
            
            channelResampler.run(evt, station, det, sampling_rate=1 * units.GHz)
    
        eventWriter.run(evt)

nevents = eventWriter.end()

print("number of events =", nevents)








