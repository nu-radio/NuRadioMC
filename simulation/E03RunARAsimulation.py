from __future__ import absolute_import, division, print_function
import argparse
# import detector simulation modules
import NuRadioReco.modules.efieldToVoltageConverterPerChannel
import NuRadioReco.modules.ARIANNA.triggerSimulator
import NuRadioReco.modules.ARA.triggerSimulator
import NuRadioReco.modules.triggerSimulator
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelBandPassFilter
import numpy as np
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.framework.channel
from scipy import constants
from NuRadioReco.utilities import units
from NuRadioMC.simulation import simulation
import logging
logging.basicConfig(level=logging.WARNING)


def get_ARA_power_mean_rms(sampling_rate, Vrms, min_freq, max_freq):
    """
    helper function to calculate the mean and rms power of the ARA tunnel diode
    for a given Vrms, sampling rate and frequency content

    Parameters
    ----------
    sampling_rate: float
        the sampling rate e.g. 1GHz
    Vrms: float
        the RMS of noise in the time domain
    min_freq: float
        the lower bandpass frequency
    max_freq: float
        the upper bandpass frequency
    """
    triggerSimulator = NuRadioReco.modules.ARA.triggerSimulator.triggerSimulator()
    channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()

    noise = NuRadioReco.framework.channel.Channel(0)

    long_noise = channelGenericNoiseAdder.bandlimited_noise(min_freq=min_freq,
                                                            max_freq=max_freq,
                                                            n_samples=2 ** 20,
                                                            sampling_rate=sampling_rate,
                                                            amplitude=Vrms,
                                                            type='perfect_white')

    noise.set_trace(long_noise, sampling_rate)

    power_noise = triggerSimulator.tunnel_diode(noise)

    power_mean = np.mean(power_noise)
    power_rms = np.sqrt(np.mean(power_noise ** 2))
    return power_mean, power_rms


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
Vrms = sim.get_Vrms()
max_freq = sim.get_bandwidth()
sampling_rate = sim.get_sampling_rate()
min_freq = 80 * units.MHz
power_mean, power_rms = get_ARA_power_mean_rms(sampling_rate, Vrms, min_freq, max_freq)

# initialize detector sim modules
efieldToVoltageConverterPerChannel = NuRadioReco.modules.efieldToVoltageConverterPerChannel.efieldToVoltageConverterPerChannel()
efieldToVoltageConverterPerChannel.begin(debug=False)
triggerSimulator = NuRadioReco.modules.triggerSimulator.triggerSimulator()
triggerSimulatorARIANNA = NuRadioReco.modules.ARIANNA.triggerSimulator.triggerSimulator()
triggerSimulatorARA = NuRadioReco.modules.ARA.triggerSimulator.triggerSimulator()
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()


def detector_simulation_ARA(evt, station, det, dt, Vrms):
    # start detector simulation
    efieldToVoltageConverterPerChannel.run(evt, station, det)  # convolve efield with antenna pattern
    # downsample trace back to detector sampling rate
    channelResampler.run(evt, station, det, sampling_rate=1. / dt)
    # bandpass filter trace, the upper bound is higher then the sampling rate which makes it just a highpass filter
    channelBandPassFilter.run(evt, station, det, passband=[min_freq, 2 * units.GHz],
                              filter_type='butter10')
    triggerSimulatorARA.run(evt, station, det,
                            power_threshold=6.5,
                            coinc_window=110 * units.ns,
                            number_concidences=3,
                            triggered_channels=[0, 1, 2, 3, 4, 5, 6, 7],
                            power_mean=power_mean, power_rms=power_rms)


sim.run(detector_simulation=detector_simulation_ARA)

