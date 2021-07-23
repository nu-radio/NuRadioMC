from __future__ import absolute_import, division, print_function
import argparse
import datetime
# import detector simulation modules
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.triggerTimeAdjuster
import NuRadioReco.modules.ARA.hardwareResponseIncorporator
import NuRadioReco.modules.trigger.highLowThreshold
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.channelBandPassFilter
from NuRadioReco.utilities import units
from NuRadioMC.simulation import simulation
import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("runstrawman")

# initialize detector sim modules
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin()
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
hardwareResponseIncorporator = NuRadioReco.modules.ARA.hardwareResponseIncorporator.hardwareResponseIncorporator()
triggerSimulator = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
triggerTimeAdjuster = NuRadioReco.modules.triggerTimeAdjuster.triggerTimeAdjuster()
triggerSimulator.begin(log_level=logging.WARNING)

#from tutorial simulation
simpleThreshold = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()
highLowThreshold = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()

class mySimulation(simulation.simulation):
    
    def _detector_simulation_filter_amp(self, evt, station, det):
        hardwareResponseIncorporator.run(evt, station, det, sim_to_data=False)
          
    '''
    def _detector_simulation_filter_amp(self, evt, station, det):
            channelBandPassFilter.run(evt, station, det, passband=[0 * units.MHz, 100000 * units.GHz],
                                      filter_type='butter', order=2)
    '''

    def _detector_simulation_trigger(self, evt, station, det):
        triggerSimulator.run(evt, station, det,
                           threshold_high=0.5* units.mV,
                           threshold_low=-0.5*units.mV,
                           number_concidences=1,
                           triggered_channels=range(16))
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
                            log_level=logging.WARNING,
                            evt_time=datetime.datetime(2018, 12, 30),
                            file_overwrite=True)
sim.run()

