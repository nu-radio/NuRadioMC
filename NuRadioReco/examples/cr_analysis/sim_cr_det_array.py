import numpy as np
import NuRadioReco.examples.cr_efficiency_analysis.helper_cr_eff as hcr
from NuRadioReco.utilities import units
from NuRadioReco.detector.detector import Detector
from NuRadioReco.framework.trigger import RadiantAUXTrigger
import NuRadioReco.modules.io.coreas.readCoREASDetector
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.channelGalacticNoiseAdder_fast
import NuRadioReco.modules.trigger.radiant_aux_trigger
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.eventTypeIdentifier
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.electricFieldResampler
import NuRadioReco.modules.io.eventWriter
import logging
import argparse

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

parser = argparse.ArgumentParser(description='Run air shower Reconstruction')

parser.add_argument('--input_file', type=str, nargs='?', default='example_data/example_data.hdf5')
parser.add_argument('--det_file', type=str, nargs='?', default='../detector/RNO_G/RNO_season_2023.json')
args = parser.parse_args()
logger.info(f'Using detector file {args.det_file} on {args.input_file}')

det = Detector(json_filename=args.det_file, antenna_by_depth=False)

#core_positions = NuRadioReco.modules.io.coreas.readCoREASDetector.get_random_core_positions()

# module to read the CoREAS file and convert it to NuRadioReco event for an array of detector stations.
readCoREASDetector = NuRadioReco.modules.io.coreas.readCoREASDetector.readCoREASDetector()
readCoREASDetector.begin(args.input_file, log_level=logging.WARNING)

# module to set the event type, if cosmic ray, the refraction of the emission on the air ice interface is taken into account
eventTypeIdentifier = NuRadioReco.modules.eventTypeIdentifier.eventTypeIdentifier()

# module to convolves the electric field with the antenna response
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin(debug=False)

# module to add the detector response, e.g. amplifier, filter, etc.
hardwareResponseIncorporator = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()

# module to add thermal noise to the channels
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
channelGenericNoiseAdder.begin()

# module to add galactic noise to the channels
channelGalacticNoiseAdder = NuRadioReco.modules.channelGalacticNoiseAdder_fast.channelGalacticNoiseAdder()
channelGalacticNoiseAdder.begin(skymodel='gsm2016', n_side=4, freq_range=np.array([0.07, 0.81]))

# module to simulate the trigger
triggerSimulator = NuRadioReco.modules.trigger.radiant_aux_trigger.triggerSimulator()
triggerSimulator.begin()

# module to filter the channels
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelBandPassFilter.begin()

# adjusts sampling rate for electirc field
electricFieldResampler = NuRadioReco.modules.electricFieldResampler.electricFieldResampler()
electricFieldResampler.begin()

# adjusts sampling rate for channels
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelResampler.begin()

# module to write the events to a .nur file
eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
eventWriter.begin(f'cr_reco_array.nur')

for evt in readCoREASDetector.run(det):
    for sta in evt.get_stations():        
        eventTypeIdentifier.run(evt, sta, "forced", 'cosmic_ray')

        efieldToVoltageConverter.run(evt, sta, det)

        channelGenericNoiseAdder.run(evt, sta, det, amplitude=1.091242302378349e-05, min_freq=80*units.MHz, max_freq=800*units.MHz, type='rayleigh')
        channelGalacticNoiseAdder.run(evt, sta, det, passband=[0.08, 0.8])

        hardwareResponseIncorporator.run(evt, sta, det, sim_to_data=True)
        
        triggerSimulator.run(evt, sta, det,
                                threshold_sigma=10,
                                triggered_channels=[13, 16, 19], #surface channels
                                trigger_name=f'radiant_trigger_10sigma')

        channelResampler.run(evt, sta, det, sampling_rate=3.2)

        electricFieldResampler.run(evt, sta, det, sampling_rate=3.2)

        eventWriter.run(evt, det=det, mode={
                    'Channels': True,
                    'ElectricFields': True,
                    'SimChannels': False,
                    'SimElectricFields': False
                })
nevents = eventWriter.end()