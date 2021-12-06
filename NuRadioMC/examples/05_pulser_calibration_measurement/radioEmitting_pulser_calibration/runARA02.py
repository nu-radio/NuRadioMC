from __future__ import absolute_import, division, print_function
import argparse
import NuRadioReco.modules.triggerTimeAdjuster
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.trigger.highLowThreshold
import NuRadioReco.modules.trigger.powerIntegration
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.custom.deltaT.calculateAmplitudePerRaySolution
from NuRadioReco.utilities import units
from NuRadioMC.simulation import simulation as simulation
import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("runARA02")
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin()
simpleThreshold = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()
powerIntegration = NuRadioReco.modules.trigger.powerIntegration.triggerSimulator()
highLowThreshold = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
electricFieldResampler = NuRadioReco.modules.electricFieldResampler.electricFieldResampler()
calculateAmplitudePerRaySolution = NuRadioReco.modules.custom.deltaT.calculateAmplitudePerRaySolution.calculateAmplitudePerRaySolution()
triggerSimulator = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
triggerTimeAdjuster = NuRadioReco.modules.triggerTimeAdjuster.triggerTimeAdjuster()
triggerSimulator.begin(log_level=logging.WARNING)


class mySimulation(simulation.simulation):

    def _detector_simulation_filter_amp(self, evt, station, det):
        # bandpass filter trace, the upper bound is higher then the sampling rate which makes it just a highpass filter
        channelBandPassFilter.run(evt, station, det, passband=[80 * units.MHz, 1000 * units.GHz],
                                  filter_type='butter', order=2)
        channelBandPassFilter.run(evt, station, det, passband=[0, 500 * units.MHz],
                                  filter_type='butter', order=10)

    def _detector_simulation_trigger(self, evt, station, det):
        # save the amplitudes to output hdf5 file
        # save amplitudes per ray tracing solution to hdf5 data output
        calculateAmplitudePerRaySolution.run(self._evt, self._station, self._det)
        triggerSimulator.run(evt, station, det,
                           threshold_high=1e-6 * self._Vrms,
                           threshold_low=-1e-6 * self._Vrms,
                           high_low_window=50 * units.ns,
                           coinc_window=170 * units.ns,
                           number_concidences=4,
                           trigger_name='highlow_2sigma_Vpol',
                           triggered_channels=[0, 1, 2, 3, 4, 5, 6, 7])

        triggerSimulator.run(evt, station, det,
                           threshold_high=1e-6 * self._Vrms,
                           threshold_low=-1e-6 * self._Vrms,
                           high_low_window=50 * units.ns,
                           coinc_window=170 * units.ns,
                           number_concidences=4,
                           trigger_name='highlow_2sigma_Hpol',
                           triggered_channels=[8, 9, 10, 11, 12, 13, 14, 15])

        triggerTimeAdjuster.run(evt, station, det)


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
