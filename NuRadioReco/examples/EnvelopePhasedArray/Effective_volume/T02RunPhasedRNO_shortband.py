"""
This file runs a phased array trigger simulation. The phased array configuration
in this file is similar to one of the proposed ideas for RNO-G: 3 GS/s, 4 antennas
at a depth of ~100 m, 30 primary phasing directions. In order to run, we need
a detector file and a configuration file, included in this folder. To run
the code, type:

python T02RunPhasedRNO_shortband.py input_neutrino_file.hdf5 proposalcompact_100m_1.5GHz.json
config_RNO.yaml output_NuRadioMC_file.hdf5 output_NuRadioReco_file.nur

The antenna positions can be changed in the detector position. The config file
defines de bandwidth for the noise RMS calculation. The properties of the phased
array can be changed in the current file - phasing angles, triggering channels,
bandpass filter and so on.

This file calculates the effective volume for the (140-700) MHz band

WARNING: this file needs NuRadioMC to be run.
"""

from __future__ import absolute_import, division, print_function
import argparse
# import detector simulation modules
import NuRadioReco.modules.phasedarray.analogBeamformedEnvelopeTrigger
import NuRadioReco.modules.channelBandPassFilter
from NuRadioReco.utilities import units
from NuRadioMC.simulation import simulation
import NuRadioReco.utilities.diodeSimulator
from NuRadioReco.utilities.diodeSimulator import get_window_around_maximum
import numpy as np

# initialize detector sim modules
triggerSimulator = NuRadioReco.modules.phasedarray.analogBeamformedEnvelopeTrigger.AnalogBeamformedEnvelopeTrigger()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()

main_low_angle = -50 * units.deg
main_high_angle = 50 * units.deg
phasing_angles = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 30))

diode_passband = (None, 200 * units.MHz)
diodeSimulator = NuRadioReco.utilities.diodeSimulator.diodeSimulator(diode_passband)


class mySimulation(simulation.simulation):

    def _detector_simulation_filter_amp(self, evt, station, det):
        channelBandPassFilter.run(evt, station, det, passband=[132 * units.MHz, 1150 * units.MHz],
                                  filter_type='butter', order=8)
        channelBandPassFilter.run(evt, station, det, passband=[0, 700 * units.MHz],
                                  filter_type='butter', order=10)

    def _detector_simulation_trigger(self, evt, station, det):

        new_sampling_rate = 3 * units.GHz

        cut_times = get_window_around_maximum(station, diodeSimulator, ratio=0.01)

        # for running an ARA-like trigger, we need to know the filtered noise
        # parameters. REMEMBER TO CHANGE THE FREQUENCIES FOR CALCULATING THEM
        power_mean, power_std = diodeSimulator.calculate_noise_parameters(new_sampling_rate,
                                                                          132 * units.MHz,
                                                                          700 * units.MHz,
                                                                          amplitude=self._Vrms)

        # first run a simple threshold trigger
        triggerSimulator.run(evt, station, det,
                             threshold_factor=3.9,  # see envelope phased trigger module for explanation
                             power_mean=power_mean,
                             power_std=power_std,
                             triggered_channels=None,  # run trigger on all channels
                             trigger_name='envelope_phasing',  # the name of the trigger
                             phasing_angles=phasing_angles,
                             ref_index=1.75,
                             output_passband=diode_passband,
                             cut_times=cut_times)


parser = argparse.ArgumentParser(description='Run NuRadioMC simulation')
parser.add_argument('--inputfilename', type=str,
                    help='path to NuRadioMC input event list', default='0.00_12_00_1.00e+16_1.00e+19.hdf5')
parser.add_argument('--detectordescription', type=str,
                    help='path to file containing the detector description', default='4antennas_100m_1.5GHz.json')
parser.add_argument('--config', type=str,
                    help='NuRadioMC yaml config file', default='config_RNO.yaml')
parser.add_argument('--outputfilename', type=str,
                    help='hdf5 output filename', default='output_PA_RNO.hdf5')
parser.add_argument('--outputfilenameNuRadioReco', type=str, nargs='?', default=None,
                    help='outputfilename of NuRadioReco detector sim file')
args = parser.parse_args()

sim = mySimulation(
    inputfilename=args.inputfilename,
    outputfilename=args.outputfilename,
    detectorfile=args.detectordescription,
    outputfilenameNuRadioReco=args.outputfilenameNuRadioReco,
    config_file=args.config)
sim.run()
