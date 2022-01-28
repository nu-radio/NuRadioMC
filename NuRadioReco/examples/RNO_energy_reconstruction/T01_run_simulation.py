from __future__ import absolute_import, division, print_function
import argparse
import NuRadioReco.modules.trigger.highLowThreshold
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
import NuRadioReco.modules.channelGenericNoiseAdder
from NuRadioReco.utilities import units
from NuRadioMC.simulation import simulation
import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("runstrawman")

parser = argparse.ArgumentParser(
    description='We start by creating some data to do reconstruction on, but feel free to use your own simulations!'
)
parser.add_argument('input_file', type=str)
parser.add_argument('--output_file', type=str, default='simulated_events.nur')
parser.add_argument('--detector_file', type=str, default='../../detector/RNO_G/RNO_single_station.json')
parser.add_argument('--config_file', type=str, default='config.yaml')
parser.add_argument('--noise_level', type=float, default=10.)

args = parser.parse_args()

# initialize detector sim modules
highLowThreshold = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
hardware_response = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()
noise_adder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
noise_adder.begin()
noise_level = args.noise_level * units.mV


class mySimulation(simulation.simulation):

    def _detector_simulation_filter_amp(self, evt, station, det):
        hardware_response.run(evt, station, det, sim_to_data=True)

    def _detector_simulation_trigger(self, evt, station, det):
        highLowThreshold.run(evt, station, det,
                                    threshold_high=20. * units.mV,
                                    threshold_low=-20. * units.mV,
                                    triggered_channels=[0, 1],  # select the LPDA channels
                                    number_concidences=2,  # 2/4 majority logic
                                    trigger_name='main_trigger'
                             )


sim = mySimulation(
    inputfilename=args.input_file,
    outputfilename='output.hdf5',
    detectorfile=args.detector_file,
    outputfilenameNuRadioReco=args.output_file,
    config_file=args.config_file,
    file_overwrite=True,
    write_detector=False
)
sim.run()
