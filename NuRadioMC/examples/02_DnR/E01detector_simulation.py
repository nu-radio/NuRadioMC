from __future__ import absolute_import, division, print_function
import argparse
from six import iteritems
# import detector simulation modules
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelGenericNoiseAdder

from NuRadioReco.utilities import units
from NuRadioMC.simulation import simulation
from NuRadioReco.framework.parameters import electricFieldParameters as efp
import logging
import numpy as np
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("runDeltaTStudy")

# initialize detector sim modules
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin(pre_pulse_time=0 * units.ns, post_pulse_time=0 * units.ns)
triggerSimulator = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()


class mySimulation(simulation.simulation):

    def _detector_simulation(self):

        # start detector simulation
        efieldToVoltageConverter.run(self._evt, self._station, self._det)  # convolve efield with antenna pattern

        # downsample trace to internal simulation sampling rate (the efieldToVoltageConverter upsamples the trace to
        # 20 GHz by default to achive a good time resolution when the two signals from the two signal paths are added)
        channelResampler.run(self._evt, self._station, self._det, sampling_rate=1. / self._dt)

        if self._is_simulate_noise():
            max_freq = 0.5 / self._dt
            norm = self._get_noise_normalization(self._station.get_id())  # assuming the same noise level for all stations
            Vrms = self._Vrms / (norm / (max_freq)) ** 0.5  # normalize noise level to the bandwidth its generated for
            channelGenericNoiseAdder.run(self._evt, self._station, self._det, amplitude=Vrms, min_freq=0 * units.MHz,
                                         max_freq=max_freq, type='rayleigh')

        # bandpass filter trace, the upper bound is higher then the sampling rate which makes it just a highpass filter
        channelBandPassFilter.run(self._evt, self._station, self._det, passband=[80 * units.MHz, 1000 * units.GHz],
                                  filter_type='butter', order=2)
        channelBandPassFilter.run(self._evt, self._station, self._det, passband=[0, 500 * units.MHz],
                                  filter_type='butter', order=10)
        triggerSimulator.run(self._evt, self._station, self._det,
                             threshold=2 * self._Vrms,
                             triggered_channels=None,
                             number_concidences=1,
                             trigger_name='pre_trigger_2sigma')

        # downsample trace back to detector sampling rate
        channelResampler.run(self._evt, self._station, self._det, sampling_rate=self._sampling_rate_detector)


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

sim = mySimulation(inputfilename=args.inputfilename,
                            outputfilename=args.outputfilename,
                            detectorfile=args.detectordescription,
                            outputfilenameNuRadioReco=args.outputfilenameNuRadioReco,
                            config_file=args.config,
                            file_overwrite=True)
sim.run()
