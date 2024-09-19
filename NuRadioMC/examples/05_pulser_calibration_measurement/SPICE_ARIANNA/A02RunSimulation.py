from __future__ import absolute_import, division, print_function
import argparse
import datetime
# import detector simulation modules
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.triggerTimeAdjuster
import NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator
import NuRadioReco.modules.trigger.highLowThreshold
from NuRadioReco.utilities import units
from NuRadioMC.simulation import simulation
import logging
#logging.basicConfig(level=logging.WARNING)
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("runstrawman")

# initialize detector sim modules
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin()
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
hardwareResponseIncorporator = NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator.hardwareResponseIncorporator()
triggerSimulator = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
triggerTimeAdjuster = NuRadioReco.modules.triggerTimeAdjuster.triggerTimeAdjuster()
#triggerSimulator.begin(log_level=logging.WARNING)
triggerSimulator.begin(log_level=logging.ERROR)


class mySimulation(simulation.simulation):

    def _detector_simulation_filter_amp(self, evt, station, det):
        hardwareResponseIncorporator.run(evt, station, det, sim_to_data=True)

    def _detector_simulation_trigger(self, evt, station, det):
        triggerSimulator.run(evt, station, det,
                           threshold_high=0.5 * units.mV,
                           threshold_low=-0.5 * units.mV,
                           high_low_window=5 * units.ns,
                           coinc_window=30 * units.ns,
                           number_concidences=2,
                           triggered_channels=range(4))
        triggerTimeAdjuster.run(evt, station, det)

"""
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
"""

if __name__ == "__main__":
    sim = mySimulation(inputfilename='SPICE_drop_event_list.hdf5',
                                outputfilename='output_MC.hdf5',
                                detectorfile='detector_db.json',
                                outputfilenameNuRadioReco='output_reco.nur',
                                config_file='config_spice.yaml',
                                #log_level=logging.WARNING,
                                log_level=logging.INFO,
                                evt_time=datetime.datetime(2019, 12, 30),
                                file_overwrite=True)
    sim.run()

