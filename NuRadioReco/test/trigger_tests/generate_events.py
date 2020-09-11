#!/usr/bin/env python
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.channelResampler
from NuRadioMC.simulation import simulation

# initialize detector sim modules
simpleThreshold = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()


class mySimulation(simulation.simulation):

    def _detector_simulation_filter_amp(self, evt, station, det):
        pass

    def _detector_simulation_trigger(self, evt, station, det):
        simpleThreshold.run(
            evt,
            station,
            det,
            threshold=3. * self._Vrms,
            triggered_channels=None,  # run trigger on all channels
            number_concidences=1,
            trigger_name='simple_threshold')  # the name of the trigger


sim = mySimulation(
    inputfilename='NuRadioReco/test/trigger_tests/trigger_test_eventlist.hdf5',
    outputfilename='input.hdf5',
    detectorfile='NuRadioReco/test/trigger_tests/trigger_test_detector.json',
    outputfilenameNuRadioReco='NuRadioReco/test/trigger_tests/trigger_test_input.nur',
    config_file='NuRadioReco/test/trigger_tests/config.yaml',
    file_overwrite=True)
sim.run()
