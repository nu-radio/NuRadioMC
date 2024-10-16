import numpy as np
import helper_cr_eff as hcr
from NuRadioReco.utilities import units
from NuRadioReco.detector.detector import Detector
import NuRadioReco.modules.io.coreas.readCoREASStation
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.channelGalacticNoiseAdder
import NuRadioReco.modules.trigger.highLowThreshold
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.eventTypeIdentifier
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.electricFieldResampler
import NuRadioReco.modules.io.eventWriter
import logging
import argparse

logger = logging.getLogger()
logger.setLevel(logging.INFO)


parser = argparse.ArgumentParser(description='Run air shower Reconstruction')

parser.add_argument('--detector_file', type=str, nargs='?', default='example_data/arianna_station_32.json',
                    help='choose detector with a single station for air shower simulation')
parser.add_argument('--input_file', type=str, nargs='?',
                    default='example_data/example_data.hdf5', help='hdf5 coreas file')

args = parser.parse_args()

logger.info(f"Use {args.detector_file} on file {args.input_file}")

det = Detector(json_filename=args.detector_file)
station_id = det.get_station_ids()[0]

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

# module to add thermal noise to the channels
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
channelGenericNoiseAdder.begin()

# module to add galactic noise to the channels
channelGalacticNoiseAdder = NuRadioReco.modules.channelGalacticNoiseAdder.channelGalacticNoiseAdder()
channelGalacticNoiseAdder.begin(n_side=4, interpolation_frequencies=np.arange(0.01, 0.81, 0.1))

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

        channelGalacticNoiseAdder.run(evt, sta, det)

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
                'SimChannels': False,
                'SimElectricFields': False
            })
nevents = eventWriter.end()
