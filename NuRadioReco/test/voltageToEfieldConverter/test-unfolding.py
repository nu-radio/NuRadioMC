import matplotlib.pyplot as plt
import numpy as np
from NuRadioReco.detector.detector import Detector
from NuRadioReco.modules import channelAddCableDelay, voltageToEfieldConverter
from NuRadioReco.modules.io import eventReader
from NuRadioReco.modules.RNO_G.hardwareResponseIncorporator import hardwareResponseIncorporator
from NuRadioReco.utilities import units, geometryUtilities, dataservers
from NuRadioReco.framework.base_trace import BaseTrace
from NuRadioReco.framework import parameters
import argparse
import logging
import os

hwresponder = hardwareResponseIncorporator()
cabledelayadder = channelAddCableDelay.channelAddCableDelay()
unfolder = voltageToEfieldConverter.voltageToEfieldConverter()
reader = eventReader.eventReader()
logger = logging.getLogger('NuRadioReco.test.voltageToEfieldConverter')

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(
         description=(
             "Test voltageToEfieldConverter (unfolding). "
             "Simple test that verifies that the unfolded electric field is (approximately) "
             "equal to the simulated electric field for a noiseless cosmic-ray event."
         )
    )
    parser.add_argument('--file', type=str, default=os.path.join(current_dir, 'cr-noiseless-with-delays.nur'), help="Input .nur file")
    parser.add_argument('--detector', type=str, default=os.path.join(current_dir, 'cr-detector.json'), help='Detector description')
    parser.add_argument('--channels', default=[13,16,19], help='Channels to use in the unfolding')
    args = parser.parse_args()

    hwresponder.begin()
    if not os.path.exists(args.file):
        logger.warning(f'Could not find "{args.file}", attempt to download from server...')
        try:
            dataservers.download_from_dataserver(os.path.join('github_ci', args.file.replace(current_dir, '').strip('/')), args.file, unpack_tarball=False)
        except OSError:
            raise FileNotFoundError(f"Could not find file '{args.file}' locally or on server. Check you have specified the file path correctly.")

    reader.begin(args.file)
    detector = Detector(args.detector)

    for event in reader.run():
        station = event.get_station()
        cabledelayadder.run(event, station, detector, mode='subtract')
        hwresponder.run(event, station, detector, sim_to_data=False, mingainlin=1e-3)
        unfolder.run(event, station, detector, use_channels=args.channels, use_MC_direction=True)

    sim_station = station.get_sim_station()
    sim_efield = sim_station.get_electric_fields()[0]
    rec_efield = station.get_electric_fields()[-1]
    if any(sim_efield.get_position() != rec_efield.get_position()):
         # shift reconstructed efield times to facilitate comparison
         # TODO: this only works for in-air detectors or in-ice antennas at equal depths
         #       because we ignore refraction into the ice
         dt = geometryUtilities.get_time_delay_from_direction(
              sim_station[parameters.stationParameters.zenith],
              sim_station[parameters.stationParameters.azimuth],
              rec_efield.get_position() - sim_efield.get_position(),
         )
         rec_efield.add_trace_start_time(-dt)


    fig, axs = plt.subplots(3, 4, figsize=(8, 4.5), sharex='col', sharey='col', layout='constrained')
    bandpass_kwargs = dict(passband=[80*units.MHz, 500*units.MHz], filter_type='butterabs')

    diff_trace = BaseTrace() # store the difference
    for j, efield in enumerate([sim_efield, rec_efield]):
        for i in range(2):
            axs[j, i].plot(efield.get_times(), efield.get_filtered_trace(**bandpass_kwargs)[i+1], lw=.5)
            axs[j, i+2].plot(
                efield.get_frequencies(),
                np.abs(efield.get_frequency_spectrum()[i+1]), lw=.5)

        # store the difference
        if not j:
                diff_trace.set_trace(-efield.get_trace(), efield.get_sampling_rate(), efield.get_trace_start_time())
        else:
            diff_trace.add_to_trace(efield)
            for i in range(2):
                axs[2, i].plot(diff_trace.get_times(), diff_trace.get_filtered_trace(**bandpass_kwargs)[i+1], lw=.5)
                axs[2, i+2].plot(
                    diff_trace.get_frequencies(),
                    np.abs(diff_trace.get_frequency_spectrum()[i+1]), lw=.5)

    for i in [2, 3]: # set frequency plot limits
        axs[0, i].set_xlim(0, 1000*units.MHz)

    # label the plots
    for i in [0, 1]:
        axs[i, 0].set_ylabel('Efield [V/m]')
        axs[2, i].set_xlabel('Time [ns]')
        axs[2, i+2].set_xlabel('Frequency [GHz]')
        axs[0, 2*i].set_title('eTheta')
        axs[0, 2*i+1].set_title('ePhi')

    axs[2, 0].set_ylabel('Difference [V/m]')

    plt.savefig('./test-cosmic-ray-unfolding.pdf')
    plt.close()

    # check that the difference is less than 1% of the simulated fluence
    total_diff_fluence = np.sum(diff_trace.get_filtered_trace(**bandpass_kwargs)**2)
    sim_efield_fluence = np.sum(sim_efield.get_filtered_trace(**bandpass_kwargs)**2)

    logger.warning(
         'Difference between unfolded fluence and simulated fluence is {:.2f}%'.format(
              100 * total_diff_fluence / sim_efield_fluence
         ))
    assert total_diff_fluence < 0.01 * sim_efield_fluence, f"Unfolded efield fluence differs by more than 1%!"
