"""
This file runs a phased array trigger simulation. The phased array configuration
in this file is inspired by the deployed ARA phased array: 1.5 GS/s, 8 antennas
at a depth of ~50 m, 15 phasing directions with primary and secondary beams.
In order to run, we need a detector file and a configuration file, included in
this folder. To run the code, type:

python T02RunPhasedARA.py input_neutrino_file.hdf5 proposalcompact_50m_1.5GHz.json
config_ARA.yaml output_NuRadioMC_file.hdf5 output_NuRadioReco_file.nur

The antenna positions can be changed in the detector position. The config file
defines de bandwidth for the noise RMS calculation. The properties of the phased
array can be changed in the current file - phasing angles, triggering channels,
bandpass filter and so on.

WARNING: this file needs NuRadioMC to be run.
"""

from __future__ import absolute_import, division, print_function
import argparse
# import detector simulation modules
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.phasedarray.triggerSimulator
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelGenericNoiseAdder
from NuRadioReco.utilities import units
from NuRadioMC.simulation import simulation
from NuRadioReco.utilities.traceWindows import get_window_around_maximum
import numpy as np
import logging
from NuRadioReco.modules.base import module
logger = module.setup_logger(level=logging.WARNING)

# initialize detector sim modules
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin(debug=False)
triggerSimulator = NuRadioReco.modules.phasedarray.triggerSimulator.triggerSimulator()
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
thresholdSimulator = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()

class mySimulation(simulation.simulation):

    def _detector_simulation(self):
        # start detector simulation
        efieldToVoltageConverter.run(self._evt, self._station, self._det)  # convolve efield with antenna pattern
        # downsample trace to 1.5 Gs/s
        new_sampling_rate = 1.5 * units.GHz
        channelResampler.run(self._evt, self._station, self._det, sampling_rate=new_sampling_rate)

        cut_times = get_window_around_maximum(self._station)

        # Bool for checking the noise triggering rate
        check_only_noise = False

        if check_only_noise:
            for channel in self._station.iter_channels():
                trace = channel.get_trace() * 0
                channel.set_trace(trace, sampling_rate = new_sampling_rate)

        if self._is_simulate_noise():
            max_freq = 0.5 / self._dt
            norm = self._get_noise_normalization(self._station.get_id())  # assuming the same noise level for all stations
            channelGenericNoiseAdder.run(self._evt, self._station, self._det, amplitude=self._Vrms, min_freq=0 * units.MHz,
                                         max_freq=max_freq, type='rayleigh', bandwidth=norm)

        # bandpass filter trace, the upper bound is higher then the sampling rate which makes it just a highpass filter
        channelBandPassFilter.run(self._evt, self._station, self._det, passband=[130 * units.MHz, 1000 * units.GHz],
                                  filter_type='butter', order=6)
        channelBandPassFilter.run(self._evt, self._station, self._det, passband=[0, 750 * units.MHz],
                                  filter_type='butter', order=10)

        # run the phased trigger
        triggerSimulator.run(self._evt, self._station, self._det,
                             threshold=2.5 * self._Vrms, # see phased trigger module for explanation
                             triggered_channels=None,  # run trigger on all channels
                             secondary_channels=[0,1,3,4,6,7], # secondary channels
                             trigger_name='primary_and_secondary_phasing', # the name of the trigger
                             coupled=True,
                             cut_times=cut_times)


parser = argparse.ArgumentParser(description='Run NuRadioMC simulation')
parser.add_argument('--inputfilename', type=str,
                    help='path to NuRadioMC input event list', default='0.00_12_00_1.00e+16_1.00e+19.hdf5')
parser.add_argument('--detectordescription', type=str,
                    help='path to file containing the detector description', default='proposalcompact_50m_1.5GHz.json')
parser.add_argument('--config', type=str,
                    help='NuRadioMC yaml config file', default='config_ARA.yaml')
parser.add_argument('--outputfilename', type=str,
                    help='hdf5 output filename', default='output_PA_ARA.hdf5')
parser.add_argument('--outputfilenameNuRadioReco', type=str, nargs='?', default=None,
                    help='outputfilename of NuRadioReco detector sim file')
args = parser.parse_args()

sim = mySimulation(inputfilename=args.inputfilename,
                            outputfilename=args.outputfilename,
                            detectorfile=args.detectordescription,
                            outputfilenameNuRadioReco=args.outputfilenameNuRadioReco,
                            config_file=args.config)
sim.run()
