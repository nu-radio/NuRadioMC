from NuRadioReco.utilities import units
import astropy.time
from NuRadioReco.detector.detector import Detector
import NuRadioReco.modules.io.coreas.readCoREASDetector
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
import NuRadioReco.modules.channelGenericNoiseAdder
#import NuRadioReco.modules.channelGalacticNoiseAdder_fast
import NuRadioReco.modules.trigger.highLowThreshold
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.eventTypeIdentifier
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.electricFieldResampler
import NuRadioReco.modules.io.eventWriter
import logging
import argparse
logger = logging.getLogger("NuRadioReco.sim_cr_det_array")   # Logging level is globally controlled
logger.setLevel(logging.WARNING)

"""
This script is an example of how to run the air shower reconstruction with a detector containing an array of stations, each with multiple antennas.

The input file needs to be a CoREAS hdf5 file where the coreas observers follow a star shape pattern. The output is a .nur file with the reconstructed event.
The input file is read in with readCoREASDetector module, which creates simulated events with all stations in the detector description,
the electric field is interpolated to the positions of the antennas. The core position of the shower can be set to a list of positions, 
see readCoREASDetector.run() for more details.

The other modules used are necessary to simulate the detector response, add noise, trigger the event, and write the event to a .nur file.
Please refer to the modules for more details.

Input parameters (all with a default provided)
---------------------
Command line input:
    python sim_cr_det_array.py --input_file path_to_coreas_file.hdf5 --det_file path_to_det.json

input_file: str
            path to CoREAS simulation hdf5 file
det_file: str
            path to json detector file

Output
---------------------
The output is a .nur file with the reconstructed event.

"""

parser = argparse.ArgumentParser(description='Run air shower Reconstruction')

parser.add_argument('--input_file', type=str, nargs='?', default='../example_data/greenland_starshape_32obs.hdf5', help='path to CoREAS simulation hdf5 file with star shape pattern')
parser.add_argument('--det_file', type=str, nargs='?', default='../../detector/RNO_G/RNO_cr_array.json', help='path to json detector file')
args = parser.parse_args()
logger.info(f'Using detector file {args.det_file} on {args.input_file}')

det = Detector(json_filename=args.det_file, antenna_by_depth=False)
det.update(astropy.time.Time('2025-1-1'))

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
hardwareResponseIncorporator.begin()

# module to add thermal noise to the channels
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
channelGenericNoiseAdder.begin()

# module to add galactic noise to the channels
#channelGalacticNoiseAdder = NuRadioReco.modules.channelGalacticNoiseAdder_fast.channelGalacticNoiseAdder()
#channelGalacticNoiseAdder.begin(skymodel='gsm2016', n_side=4, freq_range=np.array([0.07, 0.81]))

# module to simulate the trigger
triggerSimulator = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
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

for evt in readCoREASDetector.run(det, NuRadioReco.modules.io.coreas.readCoREASDetector.get_random_core_positions(-300, 0, 1600, 1800, 3)):
    for sta in evt.get_stations():        
        eventTypeIdentifier.run(evt, sta, "forced", 'cosmic_ray')

        efieldToVoltageConverter.run(evt, sta, det)

        channelGenericNoiseAdder.run(evt, sta, det, amplitude=1.091242302378349e-05, min_freq=80*units.MHz, max_freq=800*units.MHz, type='rayleigh')
        #channelGalacticNoiseAdder.run(evt, sta, det, passband=[0.08, 0.8])

        hardwareResponseIncorporator.run(evt, sta, det, sim_to_data=True)
        
        triggerSimulator.run(evt, sta, det,
                                threshold_high=5e-6,
                                threshold_low=-5e-6,
                                coinc_window=60,
                                number_concidences=2,
                                triggered_channels=[0, 1, 2, 3],
                                trigger_name='high_low')

        channelResampler.run(evt, sta, det, sampling_rate=3.2)

        electricFieldResampler.run(evt, sta, det, sampling_rate=3.2)

        eventWriter.run(evt, det=det, mode={
                    'Channels': True,
                    'ElectricFields': True,
                    'SimChannels': True,
                    'SimElectricFields': True
                })
nevents = eventWriter.end()