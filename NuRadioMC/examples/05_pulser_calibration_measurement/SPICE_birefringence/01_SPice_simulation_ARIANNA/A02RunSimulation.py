from __future__ import absolute_import, division, print_function
import argparse
import datetime
# import detector simulation modules
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator
import NuRadioReco.modules.trigger.highLowThreshold
from NuRadioReco.utilities import units
from NuRadioMC.simulation import simulation
import logging
logger = logging.getLogger("NuRadioMC.runstrawman")

# initialize detector sim modules
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin()
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
hardwareResponseIncorporator = NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator.hardwareResponseIncorporator()
triggerSimulator = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
triggerSimulator.begin(log_level=logging.WARNING)

class mySimulation(simulation.simulation):

    def _detector_simulation_filter_amp(self, evt, station, det):
        hardwareResponseIncorporator.run(evt, station, det, sim_to_data=True)

    def _detector_simulation_trigger(self, evt, station, det):
        triggerSimulator.run(evt, station, det,
                           threshold_high=0.03 * units.V,
                           threshold_low=-0.03 * units.V,
                           high_low_window=5 * units.ns,
                           coinc_window=30 * units.ns,
                           number_concidences=2,
                           triggered_channels=range(8))

if __name__ == "__main__":
    sim = mySimulation(inputfilename='input_spice.hdf5',
                                outputfilename='output_MC.hdf5',
                                detectorfile='arianna_SP_station51.json',
                                outputfilenameNuRadioReco='output_reco.nur',
                                config_file='config_spice.yaml',
                                #log_level=logging.WARNING,
                                log_level=logging.ERROR,
                                evt_time=datetime.datetime(2018, 12, 30),
                                file_overwrite=True)
    sim.run()

