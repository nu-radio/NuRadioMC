from __future__ import absolute_import, division, print_function
import argparse
# import detector simulation modules
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelBandPassFilter
from NuRadioReco.utilities import units
from NuRadioMC.simulation import simulation
import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("runstrawman")

# initialize detector sim modules
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin(debug=False)
triggerSimulator = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()

class mySimulation(simulation.simulation):

    def _detector_simulation(self):
        # start detector simulation
        efieldToVoltageConverter.run(self._evt, self._station, self._det)  # convolve efield with antenna pattern
        # downsample trace back to detector sampling rate
        channelResampler.run(self._evt, self._station, self._det, sampling_rate=1. / self._dt)

        """
        We use the [132; 700] MHz band, which is the one proposed for RNO. We approximate
        the filter with an 8th-order Butterworth on the lower end and a 10th-order
        Butterworth on the higher end.
        """
        channelBandPassFilter.run(self._evt, self._station, self._det, passband=[132 * units.MHz, 1150 * units.MHz],
                                  filter_type='butter', order=8)
        channelBandPassFilter.run(self._evt, self._station, self._det, passband=[0, 700 * units.MHz],
                                  filter_type='butter', order=10)

        """
        We run a simulation without noise and a simple amplitude trigger threshold.
        The threshold is chosen to be 1.5 times the noise RMS, which is a really
        ambitious and low threshold. The noise RMS is calculated in simulation.py
        and it depends on the noise temperature (config file, default in config_default.yaml)
        and the bandwidth, which is calculated automatically using the previously
        defined filters.
        """
        triggerSimulator.run(self._evt, self._station, self._det,
                             threshold=1.5 * self._Vrms,
                             triggered_channels=None,  # run trigger on all channels
                             number_concidences=1,
                             trigger_name='simple_threshold')  # the name of the trigger


parser = argparse.ArgumentParser(description='Run NuRadioMC simulation')
parser.add_argument('--inputfilename', type=str,
                    default=None,
                    help='path to NuRadioMC input event list')
parser.add_argument('--detectordescription', type=str,
                    default='RNO_dipoles.json',
                    help='path to file containing the detector description')
parser.add_argument('--config', type=str,
                    default='config.yaml',
                    help='NuRadioMC yaml config file')
parser.add_argument('--outputfilename', type=str,
                    help='hdf5 output filename')
parser.add_argument('--outputfilenameNuRadioReco', type=str, nargs='?', default=None,
                    help='outputfilename of NuRadioReco detector sim file')
args = parser.parse_args()

sim = mySimulation(inputfilename=args.inputfilename,
                   outputfilename=args.outputfilename,
                   detectorfile=args.detectordescription,
                   outputfilenameNuRadioReco=args.outputfilenameNuRadioReco,
                   config_file=args.config,
                   default_detector_station=101)

sim.run()
