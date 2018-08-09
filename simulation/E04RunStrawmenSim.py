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

# initialize detector sim modules
efieldToVoltageConverterPerChannel = NuRadioReco.modules.efieldToVoltageConverterPerChannel.efieldToVoltageConverterPerChannel()
efieldToVoltageConverterPerChannel.begin(debug=False)
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
                         number_concidences=1)
    if(station.has_triggered()):  # calculate more time consuming ARIANNA trigger only if station passes simple trigger
        triggerSimulatorARIANNA.run(evt, station, det,
                                    threshold_high=3 * Vrms,
                                    threshold_low=-3 * Vrms,
                                    triggered_channels=[0, 1, 2, 3, 4, 5, 6, 7],
                                    number_concidences=3)


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
sim.run(detector_simulation=detector_simulation)

