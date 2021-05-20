"""
his file runs a phased array trigger simulation. The phased array configuration
in this file is similar to one of the proposed ideas for RNO: 3 GS/s, 8 antennas
at a depth of ~50 m, 30 primary phasing directions. In order to run, we need
a detector file and a configuration file, included in this folder. To run
the code, type:

python T02RunPhasedRNO.py input_neutrino_file.hdf5 4antennas_100m_1.5GHz.json
config_RNO.yaml output_NuRadioMC_file.hdf5 output_NuRadioReco_file.nur

The antenna positions can be changed in the detector position. The config file
defines de bandwidth for the noise RMS calculation. The properties of the phased
array can be changed in the current file - phasing angles, triggering channels,
bandpass filter and so on.

WARNING: this file needs NuRadioMC to be run.
"""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import numpy as np
from NuRadioMC.simulation import simulation
import NuRadioReco
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.phasedarray.triggerSimulator
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelGenericNoiseAdder
from NuRadioReco.utilities import units
from NuRadioReco.modules.base import module

# 4 channel, 2x sampling, fft upsampling, 16 ns window
# 100 Hz -> 30.85
# 10 Hz -> 35.67
# 1 Hz -> 41.35

# 8 channel, 4x sampling, fft upsampling, 16 ns window
# 100 Hz -> 62.15
# 10 Hz -> 69.06
# 1 Hz -> 75.75

logger = module.setup_logger(level=logging.WARNING)

triggerSimulator = NuRadioReco.modules.phasedarray.triggerSimulator.triggerSimulator()
simpleThreshold = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()

main_low_angle = np.deg2rad(-59.55)
main_high_angle = np.deg2rad(59.55)
phasing_angles_4ant = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 11))
phasing_angles_8ant = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 21))

window_4ant = int(16 * units.ns * 0.5 * 2.0)
step_4ant = int(8 * units.ns * 0.5 * 2.0)

window_8ant = int(16 * units.ns * 0.5 * 4.0)
step_8ant = int(8 * units.ns * 0.5 * 4.0)


class mySimulation(simulation.simulation):

    def _detector_simulation_filter_amp(self, evt, station, det):

        channelBandPassFilter.run(evt, station, det, passband=[0.0 * units.MHz, 220.0 * units.MHz],
                                  filter_type='cheby1', order=7, rp=.1)
        channelBandPassFilter.run(evt, station, det, passband=[96.0 * units.MHz, 100.0 * units.GHz],
                                  filter_type='cheby1', order=4, rp=.1)

    def _detector_simulation_trigger(self, evt, station, det):

        Vrms = self._Vrms

        simpleThreshold.run(evt, station, det,
                            threshold=1.0 * Vrms,
                            triggered_channels=[4],  # run trigger on all channels
                            number_concidences=1,
                            trigger_name='dipole_1.0sigma')
        simpleThreshold.run(evt, station, det,
                            threshold=1.5 * Vrms,
                            triggered_channels=[4],  # run trigger on all channels
                            number_concidences=1,
                            trigger_name='dipole_1.5sigma')
        simpleThreshold.run(evt, station, det,
                            threshold=2.0 * Vrms,
                            triggered_channels=[4],  # run trigger on all channels
                            number_concidences=1,
                            trigger_name='dipole_2.0sigma')
        simpleThreshold.run(evt, station, det,
                            threshold=2.5 * Vrms,
                            triggered_channels=[4],  # run trigger on all channels
                            number_concidences=1,
                            trigger_name='dipole_2.5sigma')
        simpleThreshold.run(evt, station, det,
                            threshold=3.0 * Vrms,
                            triggered_channels=[4],  # run trigger on all channels
                            number_concidences=1,
                            trigger_name='dipole_3.0sigma')
        simpleThreshold.run(evt, station, det,
                            threshold=3.5 * Vrms,
                            triggered_channels=[4],  # run trigger on all channels
                            number_concidences=1,
                            trigger_name='dipole_3.5sigma')

        triggerSimulator.run(evt, station, det,
                             Vrms=Vrms,
                             threshold=30.85 * np.power(Vrms, 2.0),
                             triggered_channels=range(4),  # run trigger on all channels
                             phasing_angles=phasing_angles_4ant,
                             ref_index=1.75,
                             trigger_name='4ant_phasing_100Hz',  # the name of the trigger
                             trigger_adc=False,  # Don't have a seperate ADC for the trigger
                             adc_output='voltage',  # output in volts
                             trigger_filter=None,
                             upsampling_factor=2,
                             window=window_4ant,
                             step=step_4ant)


parser = argparse.ArgumentParser(description='Run NuRadioMC simulation')
parser.add_argument('--inputfilename', type=str,
                    help='path to NuRadioMC input event list', default='0.00_12_00_1.00e+16_1.00e+19.hdf5')
parser.add_argument('--detectordescription', type=str,
                    help='path to file containing the detector description', default='4antennas_100m_0.5GHz.json')
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
