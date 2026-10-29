import argparse
import os
import logging
import numpy as np

from NuRadioReco.detector.detector import Detector
from NuRadioReco.modules import channelAddCableDelay, impulsiveSignalReconstructor, channelGenericNoiseAdder
from NuRadioReco.modules.io import eventReader, eventWriter
from NuRadioReco.utilities import units, dataservers
from NuRadioReco.framework import parameters

cabledelayadder = channelAddCableDelay.channelAddCableDelay()
reconstructor = impulsiveSignalReconstructor.ImpulsiveSignalReconstructor()
noiseadder = channelGenericNoiseAdder.channelGenericNoiseAdder()
reader = eventReader.eventReader()
writer = eventWriter.eventWriter()

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)
    parser = argparse.ArgumentParser(
         description=(
             "Test ImpulsiveSignalReconstructor (simple direction reconstruction for impulsive signals) "
            )
    )
    parser.add_argument('--file', type=str, default=os.path.join(parent_dir, 'data', 'cr-noiseless-with-delays.nur'), help="Input .nur file")
    parser.add_argument(
        '--detector', type=str,
        default=os.path.join(parent_dir, 'voltageToEfieldConverter', 'cr-detector.json'),
        help='Detector description')
    parser.add_argument('--channels', default=[13,16,19], type=int, nargs='+', help='Channels to use in the reconstruction')
    parser.add_argument('--add-noise', default=False, const=True, action='store_const', help='Add noise before reconstructing')
    parser.add_argument(
        '--method', default='stft', type=str,
        help="Which method to use for the impulsive signal reconstruction. Options are 'stft', 'xcorr', or 'simple_threshold'")
    parser.add_argument('--output', type=str, default='', help="Store output in .nur file (if not provided, no output is saved).")

    args = parser.parse_args()

    ### Attempt to automatically download test file: not relevant if adapting this to your own script! ###
    if not os.path.exists(args.file):
        logger.warning(f'Could not find "{args.file}", attempt to download from server...')
        try:
            dataservers.download_from_dataserver(os.path.join('github_ci', os.path.basename(args.file)), args.file, unpack_tarball=False)
        except OSError:
            raise FileNotFoundError(f"Could not find file '{args.file}' locally or on server. Check you have specified the file path correctly.")
    ### end of download block ###

    reader.begin(args.file)
    if args.output:
        writer.begin(args.output)
    detector = Detector(args.detector)
    reconstructor.begin()
    noiseadder.begin(seed=1234)

    for event in reader.run():
        station = event.get_station()

        if args.add_noise:
            noiseadder.run(
                event, station, detector, amplitude=14*units.mV, type='rayleigh')

        # remove cable delays
        cabledelayadder.run(event, station, detector, mode='subtract')

        # run reconstruction algorithm
        reconstructor.run(
            event, station, detector,
            use_channels=args.channels, method=args.method, n_index=1.3,
        )

        if args.output:
            writer.run(event)

        sim_shower = event.get_first_sim_shower()
        if sim_shower is not None:
            simulated_zenith = sim_shower[parameters.showerParameters.zenith]
            simulated_azimuth = sim_shower[parameters.showerParameters.azimuth]
        else:
            simulated_zenith = np.nan
            simulated_azimuth = np.nan

        zenith = station[parameters.stationParameters.zenith]
        azimuth = station[parameters.stationParameters.azimuth]

        print(
            f"Reconstructed direction of event ({event.get_run_number()}, {event.get_id()}) with method {args.method}: "
            f" ({zenith/units.deg:.1f}, {azimuth/units.deg:.1f}) "
            f"/ simulated: ({simulated_zenith/units.deg:.1f}, {simulated_azimuth/units.deg:.1f})"
            )

    if args.output:
        writer.end()


