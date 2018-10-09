from __future__ import absolute_import, division, print_function
import argparse
# import detector simulation modules
import NuRadioReco.modules.efieldToVoltageConverterPerChannel
import NuRadioReco.modules.ARIANNA.triggerSimulator2
import NuRadioReco.modules.triggerSimulator
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelBandPassFilter
from NuRadioReco.utilities import units
from NuRadioMC.simulation import simulation2 as simulation
import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("runstrawman")

# initialize detector sim modules
efieldToVoltageConverterPerChannel = NuRadioReco.modules.efieldToVoltageConverterPerChannel.efieldToVoltageConverterPerChannel()
efieldToVoltageConverterPerChannel.begin(debug=False, time_resolution=1*units.ns)
triggerSimulator = NuRadioReco.modules.triggerSimulator.triggerSimulator()
triggerSimulatorARIANNA = NuRadioReco.modules.ARIANNA.triggerSimulator2.triggerSimulator()
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()

class mySimulation(simulation.simulation):


    def _detector_simulation(self):
        # start detector simulation
        efieldToVoltageConverterPerChannel.run(self._evt, self._station, self._det)  # convolve efield with antenna pattern
        # downsample trace back to detector sampling rate
        channelResampler.run(self._evt, self._station, self._det, sampling_rate=1. / self._dt)
        # bandpass filter trace, the upper bound is higher then the sampling rate which makes it just a highpass filter
                # bandpass filter trace, the upper bound is higher then the sampling rate which makes it just a highpass filter
        channelBandPassFilter.run(self._evt, self._station, self._det, passband=[80 * units.MHz, 1000 * units.GHz],
                                  filter_type='butter', order=2)
        channelBandPassFilter.run(self._evt, self._station, self._det, passband=[0, 500 * units.MHz],
                                  filter_type='butter', order=10)
        # first run a simple threshold trigger
        triggerSimulator.run(self._evt, self._station, self._det,
                             threshold=3 * self._Vrms,
                             triggered_channels=None,  # run trigger on all channels
                             number_concidences=1,
                             trigger_name='simple_threshold')  # the name of the trigger
        
        # run a high/low trigger on the 4 downward pointing LPDAs
        triggerSimulatorARIANNA.run(self._evt, self._station, self._det,
                                    threshold_high=4 * self._Vrms,
                                    threshold_low=-4 * self._Vrms,
                                    triggered_channels=[0, 1, 2, 3],  # select the LPDA channels
                                    number_concidences=2,  # 2/4 majority logic
                                    cut_trace=False,
                                    trigger_name='LPDA_2of4_4.1sigma',
                                    set_not_triggered=(not self._station.has_triggered("simple_threshold"))) # calculate more time consuming ARIANNA trigger only if station passes simple trigger
        
        # run a high/low trigger on the 4 surface dipoles 
        triggerSimulatorARIANNA.run(self._evt, self._station, self._det,
                                    threshold_high=3 * self._Vrms,
                                    threshold_low=-3 * self._Vrms,
                                    triggered_channels=[4, 5, 6, 7], # select the bicone channels
                                    number_concidences=4, # 4/4 majority logic
                                    cut_trace=False,
                                    trigger_name='surface_dipoles_4of4_3sigma',
                                    set_not_triggered=(not self._station.has_triggered("simple_threshold"))) # calculate more time consuming ARIANNA trigger only if station passes simple trigger


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

sim = mySimulation(eventlist=args.inputfilename,
                            outputfilename=args.outputfilename,
                            detectorfile=args.detectordescription,
                            station_id=101,
                            outputfilenameNuRadioReco=args.outputfilenameNuRadioReco,
                            config_file=args.config)
sim.run()

