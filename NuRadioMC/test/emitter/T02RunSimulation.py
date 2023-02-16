#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import argparse
# import detector simulation modules
import NuRadioReco.modules.trigger.highLowThreshold
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.channelBandPassFilter
from NuRadioReco.utilities import units
from NuRadioMC.simulation import simulation
import logging
import os
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("runstrawman")

# initialize detector sim modules
triggerSimulatorHighLow = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
triggerSimulatorSimple = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()


class mySimulation(simulation.simulation):

    def _detector_simulation_filter_amp(self, evt, station, det):
        # bandpass filter trace, the upper bound is higher then the sampling rate which makes it just a highpass filter
        channelBandPassFilter.run(evt, station, det, passband=[80 * units.MHz, 1000 * units.GHz],
                                  filter_type='butter', order=2)
        channelBandPassFilter.run(evt, station, det, passband=[0, 500 * units.MHz],
                                  filter_type='butter', order=10)

    def _detector_simulation_trigger(self, evt, station, det):
        # run a high/low trigger on the 4 downward pointing LPDAs
        triggerSimulatorHighLow.run(evt, station, det,
                                    threshold_high=2 * self._Vrms,
                                    threshold_low=-2 * self._Vrms,
                                    triggered_channels=None,  # select the LPDA channels
                                    number_concidences=1,  # 2/4 majority logic
                                    trigger_name='highlow_2sigma')


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
                    config_file=os.path.join(path, args.config),
                    file_overwrite=True,
                    log_level=logging.DEBUG)
sim.run()

