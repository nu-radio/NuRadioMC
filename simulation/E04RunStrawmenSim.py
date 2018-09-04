from __future__ import absolute_import, division, print_function
import argparse
# import detector simulation modules
import NuRadioReco.modules.efieldToVoltageConverterPerChannel
import NuRadioReco.modules.ARIANNA.triggerSimulator
import NuRadioReco.modules.triggerSimulator
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelBandPassFilter
from NuRadioReco.utilities import units
from NuRadioMC.simulation import simulation
import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("runstrawman")

# initialize detector sim modules
efieldToVoltageConverterPerChannel = NuRadioReco.modules.efieldToVoltageConverterPerChannel.efieldToVoltageConverterPerChannel()
efieldToVoltageConverterPerChannel.begin(debug=False, time_resolution=1*units.ns)
triggerSimulator = NuRadioReco.modules.triggerSimulator.triggerSimulator()
triggerSimulatorARIANNA = NuRadioReco.modules.ARIANNA.triggerSimulator.triggerSimulator()
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()


def detector_simulation(evt, station, det, dt, Vrms):
    # start detector simulation
    efieldToVoltageConverterPerChannel.run(evt, station, det)  # convolve efield with antenna pattern
    # downsample trace back to detector sampling rate
    channelResampler.run(evt, station, det, sampling_rate=1. / dt)
    # bandpass filter trace, the upper bound is higher then the sampling rate which makes it just a highpass filter
    channelBandPassFilter.run(evt, station, det, passband=[80 * units.MHz, 1000 * units.MHz],
                              filter_type='butter10')
    triggerSimulator.run(evt, station, det,
                        threshold=3 * Vrms,
#                         triggered_channels=None,
                        triggered_channels=[0, 1, 2, 3, 4, 5, 6, 7],
#                          triggered_channels=[0, 1, 2, 3],
                         number_concidences=1,
                         trigger_name='simple_threshold')
    triggerSimulatorARIANNA.run(evt, station, det,
                                threshold_high=3 * Vrms,
                                threshold_low=-3 * Vrms,
                                triggered_channels=[0, 1, 2, 3, 4, 5, 6, 7],
                                number_concidences=3,
                                cut_trace=False,
                                trigger_name='high_low_3of8',
                                set_not_triggered=(not station.has_triggered("simple_threshold"))) # calculate more time consuming ARIANNA trigger only if station passes simple trigger
    triggerSimulatorARIANNA.run(evt, station, det,
                                threshold_high=3 * Vrms,
                                threshold_low=-3 * Vrms,
                                triggered_channels=[0, 1, 2, 3, 4, 5, 6, 7],
                                number_concidences=2,
                                cut_trace=False,
                                trigger_name='high_low_2of8',
                                set_not_triggered=(not station.has_triggered("simple_threshold"))) # calculate more time consuming ARIANNA trigger only if station passes simple trigger
    triggerSimulatorARIANNA.run(evt, station, det,
                                threshold_high=3 * Vrms,
                                threshold_low=-3 * Vrms,
                                triggered_channels=[0, 1, 2, 3, 12, 13, 14, 15],
                                number_concidences=3,
                                cut_trace=False,
                                trigger_name='high_low_2of8_LPDAs',
                                set_not_triggered=(not station.has_triggered("simple_threshold"))) # calculate more time consuming ARIANNA trigger only if station passes simple trigger
    triggerSimulatorARIANNA.run(evt, station, det,
                                threshold_high=3 * Vrms,
                                threshold_low=-3 * Vrms,
                                triggered_channels=[0, 1, 2, 3],
                                number_concidences=2,
                                cut_trace=False,
                                trigger_name='high_low_2of4_LPDA',
                                set_not_triggered=(not station.has_triggered("simple_threshold"))) # calculate more time consuming ARIANNA trigger only if station passes simple trigger
    triggerSimulatorARIANNA.run(evt, station, det,
                                threshold_high=3 * Vrms,
                                threshold_low=-3 * Vrms,
                                triggered_channels=[4, 5, 6, 7],
                                number_concidences=4,
                                cut_trace=False,
                                trigger_name='high_low_4of4_dipoles',
                                set_not_triggered=(not station.has_triggered("simple_threshold"))) # calculate more time consuming ARIANNA trigger only if station passes simple trigger
    triggerSimulatorARIANNA.run(evt, station, det,
                                threshold_high=3 * Vrms,
                                threshold_low=-3 * Vrms,
                                triggered_channels=[8, 9, 10, 11],
                                number_concidences=2,
                                cut_trace=False,
                                trigger_name='high_low_2of4_deep_dipoles')
    triggerSimulatorARIANNA.run(evt, station, det,
                                threshold_high=3 * Vrms,
                                threshold_low=-3 * Vrms,
                                triggered_channels=[0, 1, 2, 3, 4, 5, 6, 7],
                                number_concidences=6,
                                cut_trace=False,
                                trigger_name='high_low_6of8_3sigma',
                                set_not_triggered=(not station.has_triggered("simple_threshold"))) # calculate more time consuming ARIANNA trigger only if station passes simple trigger
    triggerSimulatorARIANNA.run(evt, station, det,
                                threshold_high=4 * Vrms,
                                threshold_low=-4 * Vrms,
                                triggered_channels=[0, 1, 2, 3],
                                number_concidences=2,
                                cut_trace=False,
                                trigger_name='high_low_2of4_LPDA_4sigma',
                                set_not_triggered=(not station.has_triggered("simple_threshold"))) # calculate more time consuming ARIANNA trigger only if station passes simple trigger
    triggerSimulatorARIANNA.run(evt, station, det,
                                threshold_high=3.3 * Vrms,
                                threshold_low=-3.3 * Vrms,
                                triggered_channels=[0, 1, 2, 3],
                                number_concidences=4,
                                cut_trace=False,
                                trigger_name='high_low_4of4_LPDA_3.3sigma',
                                set_not_triggered=(not station.has_triggered("simple_threshold"))) # calculate more time consuming ARIANNA trigger only if station passes simple trigger
    triggerSimulatorARIANNA.run(evt, station, det,
                                threshold_high=3.3 * Vrms,
                                threshold_low=-3.3 * Vrms,
                                triggered_channels=[4, 5, 6, 7],
                                number_concidences=4,
                                cut_trace=False,
                                trigger_name='high_low_4of4_3.3sigma',
                                set_not_triggered=(not station.has_triggered("simple_threshold"))) # calculate more time consuming ARIANNA trigger only if station passes simple trigger




parser = argparse.ArgumentParser(description='Run NuRadioMC simulation')
parser.add_argument('inputfilename', type=str,
                    help='path to NuRadioMC input event list')
parser.add_argument('detectordescription', type=str,
                    help='path to file containing the detector description')
parser.add_argument('outputfilename', type=str,
                    help='hdf5 output filename')
parser.add_argument('outputfilenameNuRadioReco', type=str, nargs='?', default=None,
                    help='outputfilename of NuRadioReco detector sim file')
args = parser.parse_args()

sim = simulation.simulation(eventlist=args.inputfilename,
                            outputfilename=args.outputfilename,
                            detectorfile=args.detectordescription,
                            station_id=101,
                            Tnoise=350.,
                            outputfilenameNuRadioReco=args.outputfilenameNuRadioReco)
sim.run(detector_simulation=detector_simulation, number_of_triggers=11)

