import argparse
import os
import logging
import numpy as np

from NuRadioReco.detector.detector import Detector
from NuRadioReco.modules import channelAddCableDelay, crEfieldReconstructor, channelGenericNoiseAdder, cosmicRayEnergyReconstructor, electricFieldBandPassFilter
from NuRadioReco.modules.io import eventReader
from NuRadioReco.utilities import units, dataservers
from NuRadioReco.framework import parameters

cabledelayadder = channelAddCableDelay.channelAddCableDelay()
bandpassfilter = electricFieldBandPassFilter.electricFieldBandPassFilter()
reconstructor = crEfieldReconstructor.CREfieldReconstructor()
noiseadder = channelGenericNoiseAdder.channelGenericNoiseAdder()
energyreco = cosmicRayEnergyReconstructor.cosmicRayEnergyReconstructor()
reader = eventReader.eventReader()

logger = logging.getLogger()
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)
    parser = argparse.ArgumentParser(
         description=(
             "Test CREfieldReconstructor (forward-folding reconstruction of cosmic-ray electric field) "
             "and cosmicRayEnergyReconstructor (energy reconstruction from electric field)."
             "Reconstructs the electric field for a single simulated cosmic ray. "
         )
    )
    parser.add_argument('--file', type=str, default=os.path.join(parent_dir, 'data', 'cr-noiseless-with-delays.nur'), help="Input .nur file")
    parser.add_argument(
        '--detector', type=str,
        default=os.path.join(parent_dir, 'voltageToEfieldConverter', 'cr-detector.json'),
        help='Detector description')
    parser.add_argument('--channels', default=[13,16,19], help='Channels to use in the reconstruction')
    parser.add_argument('--add-noise', default=False, const=True, action='store_const', help='Add noise before reconstructing')
    parser.add_argument('--debug', default=False, const=True, action='store_const', help='Produce debug plots')

    args = parser.parse_args()

    if not os.path.exists(args.file):
        logger.warning(f'Could not find "{args.file}", attempt to download from server...')
        try:
            dataservers.download_from_dataserver(os.path.join('github_ci', os.path.basename(args.file)), args.file, unpack_tarball=False)
        except OSError:
            raise FileNotFoundError(f"Could not find file '{args.file}' locally or on server. Check you have specified the file path correctly.")

    reader.begin(args.file)
    detector = Detector(args.detector)
    reconstructor.begin(debug=args.debug, debug_folder=current_dir)
    energyreco.begin()
    noiseadder.begin(seed=1234)

    for event in reader.run():
        station = event.get_station()
        if args.add_noise:
            noiseadder.run(
                event, station, detector, amplitude=14*units.mV, type='rayleigh')

        cabledelayadder.run(event, station, detector, mode='subtract')
        reconstructor.run(event, station, detector, channel_ids=args.channels)

        # the energy reco expects the electric field to be bandpass-filtered to 80-300 MHz
        bandpassfilter.run(
            event, station, detector,
            passband=[80*units.MHz, 300*units.MHz], filter_type='butter', order=10)
        energyreco.run(event, station, detector)

        reconstructed_shower_energy = station[parameters.stationParameters.cr_energy_em]
        simulated_shower_energy = np.nan
        sim_shower = event.get_first_sim_shower()
        if sim_shower is not None:
            simulated_shower_energy = sim_shower[parameters.showerParameters.energy]

        print(
            f"Reconstructed shower energy: {reconstructed_shower_energy/units.eV:.3e} eV "
            f"/ simulated: {simulated_shower_energy/units.eV:.3e} eV"
            )

        ### relevant only for CI testing!
        if sim_shower is not None:
            if np.abs(np.log(reconstructed_shower_energy / simulated_shower_energy)) > 0.35: # 35%
                raise AssertionError("Difference between reconstructed and simulated shower energy larger than expected!")


