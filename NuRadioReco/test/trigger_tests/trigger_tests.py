#!/usr/bin/env python
import NuRadioReco.detector.detector
import NuRadioReco.modules.io.eventReader
import NuRadioReco.modules.io.eventWriter
import NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator
import NuRadioReco.modules.trigger.highLowThreshold
import NuRadioReco.modules.trigger.multiHighLowThreshold
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.phasedarray.triggerSimulator
import NuRadioReco.modules.efieldToVoltageConverter
from NuRadioReco.utilities import units

det = NuRadioReco.detector.detector.Detector('NuRadioReco/test/trigger_tests/trigger_test_detector.json')
event_reader = NuRadioReco.modules.io.eventReader.eventReader()
event_reader.begin('NuRadioReco/test/trigger_tests/trigger_test_input.nur')
event_writer = NuRadioReco.modules.io.eventWriter.eventWriter()
event_writer.begin('NuRadioReco/test/trigger_tests/trigger_test_output.nur')

high_low_trigger = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
multi_high_low_trigger = NuRadioReco.modules.trigger.multiHighLowThreshold.triggerSimulator()
simple_threshold_trigger = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()
phased_array_trigger = NuRadioReco.modules.phasedarray.triggerSimulator.triggerSimulator()
efield_to_voltage_converter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efield_to_voltage_converter.begin()
hardware_response_incorporator = NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator.hardwareResponseIncorporator()

for event in event_reader.run():
    station = event.get_station(1)
    efield_to_voltage_converter.run(event, station, det)
    hardware_response_incorporator.run(event, station, det, True)
    high_low_trigger.run(event, station, det)
    multi_high_low_trigger.run(event, station, det, trigger_name="default_multi_high_low")
    simple_threshold_trigger.run(event, station, det)
    phased_array_trigger.run(event, station, det)
    
    event_writer.run(event)

