import astropy.time
from NuRadioReco.utilities import units
from NuRadioReco.detector.detector import Detector
import NuRadioReco.modules.io.coreas.readCoREASStation
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
import NuRadioReco.modules.channelGenericNoiseAdder
#import NuRadioReco.modules.channelGalacticNoiseAdder
import NuRadioReco.modules.trigger.highLowThreshold
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.eventTypeIdentifier
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.electricFieldResampler
import NuRadioReco.modules.io.eventWriter
import logging
import argparse

logger = logging.getLogger("NuRadioReco.sim_cr_single_station")   # Logging level is globally controlled
logger.setLevel(logging.INFO)

"""
This script is an example of how to run the air shower reconstruction with a detector containing only a single station with multiple antennas. 

The input file needs to be a CoREAS hdf5 file. The output is a .nur file with the reconstructed event.
The input file is read in with readCoREASStation module, which creates simulated events for each CoREAS observer.
The other modules used are necessary to simulate the detector response, add noise, trigger the event, and write the event to a .nur file.
Please refer to the modules for more details.

Input parameters (all with a default provided)
---------------------
Command line input:
    python sim_cr_single_station.py --det_file ../example_data/arianna_station_32.json --input_file ../example_data/example_data.hdf5

detector_file: str
            path to json detector file
input_file: str
            path to CoREAS simulation hdf5 file

Output
---------------------
The output is a .nur file with the reconstructed event.

"""

parser = argparse.ArgumentParser(description='Run air shower Reconstruction')

parser.add_argument('--input_file', type=str, nargs='?',
                    default='../example_data/greenland_starshape_32obs.hdf5', help='hdf5 coreas file')
parser.add_argument('--det_file', type=str, nargs='?', default='../../detector/RNO_G/RNO_single_station.json',
                    help='choose detector with a single station for air shower simulation')

args = parser.parse_args()

logger.info(f"Use {args.det_file} on file {args.input_file}")

det = Detector(json_filename=args.det_file)
station_id = det.get_station_ids()[0]
det.update(astropy.time.Time('2025-1-1'))

# module to read the CoREAS file and convert it to NuRadioReco event, each observer is a new event with a different core position
readCoREASStation = NuRadioReco.modules.io.coreas.readCoREASStation.readCoREASStation()
readCoREASStation.begin([args.input_file], station_id, debug=False)

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
# channelGalacticNoiseAdder = NuRadioReco.modules.channelGalacticNoiseAdder.channelGalacticNoiseAdder()
# channelGalacticNoiseAdder.begin(n_side=4, interpolation_frequencies=np.arange(0.01, 0.81, 0.1))

# module to simulate the trigger
triggerSimulator = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
triggerSimulator.begin()

# module to filter the channels
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelBandPassFilter.begin()

# module adjust the sampling rate of the electric field
electricFieldResampler = NuRadioReco.modules.electricFieldResampler.electricFieldResampler()
electricFieldResampler.begin()

# module adjust the sampling rate of the channels
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelResampler.begin()

# module to write the event to a .nur file
eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
eventWriter.begin('cr_single_station.nur')

for evt in readCoREASStation.run(detector=det):
    for sta in evt.get_stations():
        eventTypeIdentifier.run(evt, sta, "forced", 'cosmic_ray')

        efieldToVoltageConverter.run(evt, sta, det)

        channelGenericNoiseAdder.run(evt, sta, det, amplitude=1.091242302378349e-05, min_freq=80*units.MHz, max_freq=800*units.MHz, type='rayleigh')

        #channelGalacticNoiseAdder.run(evt, sta, det)

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
