"""
This file runs a phased array trigger simulation. The phased array configuration
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

# 4 channel, 2x sampling
#  100 Hz -> 2.91
#  10 Hz -> 3.06
#  1 Hz -> 3.20

# Half window integration
#  100 Hz -> 3.66
#  10 Hz -> 3.88
#  1 Hz -> 4.09

# 8 channel, 4x sampling
# 100 Hz -> 4.25
# 10 Hz -> 4.56
# 1 Hz -> 4.88

# Half window integration
# 100 Hz -> 5.18
#  10 Hz -> 5.46
#  1 Hz -> 5.71


from __future__ import absolute_import, division, print_function
import argparse
import numpy as np
import logging

import sys
sys.path.append('/home/danielsmith/icecube_gen2/NuRadioReco')

import NuRadioReco
print(NuRadioReco.__file__)

from NuRadioMC.simulation import simulation

import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.phasedarray.triggerSimulator
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelGenericNoiseAdder
from NuRadioReco.utilities import units
from NuRadioReco.modules.base import module
from NuRadioReco.utilities.traceWindows import get_window_around_maximum

logger = module.setup_logger(level=logging.WARNING)

# initialize detector sim modules
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin(debug=False)

triggerSimulator = NuRadioReco.modules.phasedarray.triggerSimulator.triggerSimulator()
simpleThreshold = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()

channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()

main_low_angle = np.deg2rad(-59.54968597864437)
main_high_angle = np.deg2rad(59.54968597864437)
phasing_angles_4ant = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 11))
phasing_angles_8ant = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 21))

class mySimulation(simulation.simulation):

    def _detector_simulation_filter_amp(self, evt, station, det):
        channelBandPassFilter.run(self._evt, self._station, self._det, passband=[80 * units.MHz, 230 * units.MHz],
                                  filter_type='cheby1', order=4, rp=.1)
        channelBandPassFilter.run(self._evt, self._station, self._det, passband=[0 * units.MHz, 240 * units.MHz],
                                  filter_type='cheby1', order=9, rp=.1)
        pass

    def _detector_simulation_part2(self):

        # start detector simulation
        efieldToVoltageConverter.run(self._evt, self._station, self._det)  # convolve efield with antenna pattern

        # downsample trace back to detector sampling rate
        new_sampling_rate = 500.0 * units.MHz
        channelResampler.run(self._evt, self._station, self._det, sampling_rate=new_sampling_rate)

        Vrms = self._Vrms 

        if(self._is_simulate_noise()):
            print("Adding noise")

            min_freq = 0.0 * units.MHz
            max_freq = 250 * units.MHz
            bandwidth = 0.1732429316625746 
            Vrms_ratio = np.sqrt(bandwidth / (max_freq - min_freq))

            channelGenericNoiseAdder.run(self._evt, self._station, self._det, amplitude=Vrms / Vrms_ratio,
                                         min_freq=0 * units.MHz,
                                         max_freq=max_freq, type='rayleigh')

        simpleThreshold.run(self._evt, self._station, self._det,
                            threshold=3.5 * Vrms,
                            triggered_channels=[4],  # run trigger on all channels
                            number_concidences=1,
                            trigger_name=f'dipole_3.5sigma')
        simpleThreshold.run(self._evt, self._station, self._det,
                            threshold=3.0 * Vrms,
                            triggered_channels=[4],  # run trigger on all channels
                            number_concidences=1,
                            trigger_name=f'dipole_3.0sigma')
        simpleThreshold.run(self._evt, self._station, self._det,
                            threshold=2.5 * Vrms,
                            triggered_channels=[4],  # run trigger on all channels
                            number_concidences=1,
                            trigger_name=f'dipole_2.5sigma')
        simpleThreshold.run(self._evt, self._station, self._det,
                            threshold=2.0 * Vrms,
                            triggered_channels=[4],  # run trigger on all channels
                            number_concidences=1,
                            trigger_name=f'dipole_2.0sigma')
        simpleThreshold.run(self._evt, self._station, self._det,
                            threshold=1.5 * Vrms,
                            triggered_channels=[4],  # run trigger on all channels
                            number_concidences=1,
                            trigger_name=f'dipole_1.5sigma')
        simpleThreshold.run(self._evt, self._station, self._det,
                            threshold=1.0 * Vrms,
                            triggered_channels=[4],  # run trigger on all channels
                            number_concidences=1,
                            trigger_name=f'dipole_1.0sigma')

        ff = 500 * units.MHz

        # run the 4 phased trigger
        triggerSimulator.run(self._evt, self._station, self._det,
                             Vrms = Vrms,
                             threshold = 3.66 * Vrms,  # see phased trigger module for explanation
                             triggered_channels=range(4, 8),  # run trigger on all channels
                             phasing_angles=phasing_angles_4ant,
                             ref_index = 1.75, 
                             trigger_name='4ant_phasing',  # the name of the trigger
                             trigger_adc=False, # Don't have a seperate ADC for the trigger
                             adc_output='voltage', # output in volts
                             nyquist_zone=None, # first nyquist zone
                             bandwidth_edge=20 * units.MHz,                             
                             upsampling_factor=2,
                             window=int(16 / (ff*2) ), 
                             step = int(8  / (ff*2) ))

        # run the 8 channel phased trigger
        triggerSimulator.run(self._evt, self._station, self._det,
                             Vrms = Vrms,
                             threshold = 5.18 * Vrms,  # see phased trigger module for explanation
                             triggered_channels=None,  # run trigger on all channels
                             phasing_angles=phasing_angles_8ant, 
                             ref_index = 1.75, 
                             trigger_name='8ant_phasing',  # the name of the trigger
                             trigger_adc=False, # Don't have a seperate ADC for the trigger
                             adc_output='voltage', # output in volts
                             nyquist_zone=None, # first nyquist zone
                             bandwidth_edge=20 * units.MHz,                             
                             upsampling_factor=4,
                             window=int(16 / (ff*4) ), 
                             step = int(8  / (ff*4) )) # upsample by this amount

parser = argparse.ArgumentParser(description='Run NuRadioMC simulation')
parser.add_argument('--inputfilename', type=str, help='path to NuRadioMC input event list', default='0.00_12_00_1.00e+16_1.00e+19.hdf5')                    
parser.add_argument('--detectordescription', type=str, help='path to file containing the detector description', default='4antennas_100m_1.5GHz.json')                    
parser.add_argument('--config', type=str, help='NuRadioMC yaml config file', default='config_RNO.yaml')                    
parser.add_argument('--outputfilename', type=str, help='hdf5 output filename', default='output_PA_RNO.hdf5')                    
parser.add_argument('--outputfilenameNuRadioReco', type=str, nargs='?', default=None, help='outputfilename of NuRadioReco detector sim file')                    
args = parser.parse_args()

sim = mySimulation(
    inputfilename=args.inputfilename,
    outputfilename=args.outputfilename,
    detectorfile=args.detectordescription,
    outputfilenameNuRadioReco=args.outputfilenameNuRadioReco,
    config_file=args.config)
sim.run()
