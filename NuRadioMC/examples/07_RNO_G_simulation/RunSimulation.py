from __future__ import absolute_import, division, print_function
import argparse
import NuRadioReco.modules.trigger.highLowThreshold
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
from NuRadioReco.utilities import units
from NuRadioMC.simulation import simulation

highLowThreshold = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
hardware_response = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()


class mySimulation(simulation.simulation):

    def _detector_simulation_filter_amp(self, evt, station, det):
        hardware_response.run(evt, station, det, sim_to_data=True)

    def _detector_simulation_trigger(self, evt, station, det):
        highLowThreshold.run(
            evt,
            station,
            det,
            threshold_high=20. * units.mV,
            threshold_low=-20. * units.mV,
            triggered_channels=[0],
            number_concidences=1,
            trigger_name='main_trigger'
        )


parser = argparse.ArgumentParser(description='Run RNO-G simulation')
parser.add_argument('input_filename', type=str)
parser.add_argument('--detector_file', type=str, default='../../../NuRadioReco/detector/RNO_G/RNO_single_station.json')
parser.add_argument('--config', type=str, default='RNO_config.yaml')
parser.add_argument('--output_hdf', type=str, default='output.hdf5')
parser.add_argument('--output_nur', type=str, default='output.nur')

args = parser.parse_args()

if __name__ == "__main__":
    sim = mySimulation(
        inputfilename=args.input_filename,
        outputfilename=args.output_hdf,
        detectorfile=args.detector_file,
        outputfilenameNuRadioReco=args.output_nur,
        config_file=args.config,
        file_overwrite=True,
        write_detector=False,
        default_detector_channel=0,
        default_detector_station=11
    )
    sim.run()
