#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import argparse
# import detector simulation modules
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.trigger.highLowThreshold
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelGenericNoiseAdder
from NuRadioReco.utilities import units
from NuRadioMC.simulation import simulation
import logging
import os
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("runstrawman")

# initialize detector sim modules
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin()
triggerSimulatorHighLow = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
triggerSimulatorSimple = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()


class mySimulation(simulation.simulation):

    def _detector_simulation(self):
        # start detector simulation
        if(bool(self._cfg['signal']['zerosignal'])):
            self._increase_signal(None, 0)

        efieldToVoltageConverter.run(self._evt, self._station, self._det)  # convolve efield with antenna pattern
        # downsample trace to internal simulation sampling rate (the efieldToVoltageConverter upsamples the trace to
        # 20 GHz by default to achive a good time resolution when the two signals from the two signal paths are added)
        channelResampler.run(self._evt, self._station, self._det, sampling_rate=1. / self._dt)

        if bool(self._cfg['noise']):
            Vrms = self._Vrms / (self._bandwidth / (2.5 * units.GHz)) ** 0.5  # normalize noise level to the bandwidth its generated for
            channelGenericNoiseAdder.run(self._evt, self._station, self._det, amplitude=Vrms, min_freq=0 * units.MHz,
                                         max_freq=2.5 * units.GHz, type='rayleigh')

        # bandpass filter trace, the upper bound is higher then the sampling rate which makes it just a highpass filter
        channelBandPassFilter.run(self._evt, self._station, self._det, passband=[80 * units.MHz, 1000 * units.GHz],
                                  filter_type='butter', order=2)
        channelBandPassFilter.run(self._evt, self._station, self._det, passband=[0, 500 * units.MHz],
                                  filter_type='butter', order=10)

        # run a high/low trigger on the 4 downward pointing LPDAs
        triggerSimulatorHighLow.run(self._evt, self._station, self._det,
                                    threshold_high=2 * self._Vrms,
                                    threshold_low=-2 * self._Vrms,
                                    triggered_channels=None,  # select the LPDA channels
                                    number_concidences=1,  # 2/4 majority logic
                                    trigger_name='highlow_2sigma')

        # downsample trace back to detector sampling rate
#         channelResampler.run(self._evt, self._station, self._det, sampling_rate=self._sampling_rate_detector)


path = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='Run NuRadioMC simulation')
parser.add_argument('inputfilename', type=str,
                    help='path to NuRadioMC input event list')
parser.add_argument('detectordescription', type=str,
                    help='path to file containing the detector description')
parser.add_argument('config', type=str,
                    help='NuRadioMC yaml config file')
parser.add_argument('outputfilename', type=str,
                    help='hdf5 output filename')
parser.add_argument('outputfilenameNuRadioReco', type=str, nargs='?', default=None,
                    help='outputfilename of NuRadioReco detector sim file')
args = parser.parse_args()
outputfilenameNuRadioReco = args.outputfilenameNuRadioReco
if(outputfilenameNuRadioReco is not None):
    outputfilenameNuRadioReco = os.path.join(path, outputfilenameNuRadioReco)

sim = mySimulation(inputfilename=os.path.join(path, args.inputfilename),
                            outputfilename=os.path.join(path, args.outputfilename),
                            detectorfile=os.path.join(path, args.detectordescription),
                            outputfilenameNuRadioReco=outputfilenameNuRadioReco,
                            config_file=os.path.join(path, args.config))
sim.run()

