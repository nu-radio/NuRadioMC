"""
This file calculates the SNR curves for a given phased array configuration.
Neutrinos specified by the input file should come from a fixed arrival direction
if the SNR curve for that particular direction is required. It is convenient
that the array sees the neutrino at the Cherenkov angle, so that the signals
are clearer. The signals are then rescaled to have the appropriate SNR and
the phased trigger is run to decide whether the event is selected.

python T02RunSNR.py input_neutrino_file.hdf5 proposalcompact_50m_1.5GHz.json
config.yaml output_NuRadioMC_file.hdf5 output_SNR_file.hdf5 output_NuRadioReco_file.nur(optional)

The output_SNR_file.hdf5 contains the information on the SNR curves stored in
hdf5 format. The fields in this file are:
    - 'total_events', the total number of events that have been used
    - 'SNRs', a list with the SNR

The antenna positions can be changed in the detector position. The config file
defines de bandwidth for the noise RMS calculation. The properties of tnhe phased
array can be changed in the current file - phasing angles, triggering channels,
bandpass filter and so on.

WARNING: this file needs NuRadioMC to be run.
"""
from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import copy
import numpy as np
from NuRadioMC.simulation import simulation
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.phasedarray.triggerSimulator
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelGenericNoiseAdder
from NuRadioReco.utilities import units
from NuRadioReco.modules.base import module

parser = argparse.ArgumentParser(description='Run NuRadioMC simulation')
parser.add_argument('--inputfilename', type=str,
                    help='path to NuRadioMC input event list')
parser.add_argument('--detectordescription', type=str,
                    help='path to file containing the detector description')
parser.add_argument('--config', type=str,
                    help='NuRadioMC yaml config file')
parser.add_argument('--outputfilename', type=str,
                    help='hdf5 output filename')
parser.add_argument('--outputSNR', type=str,
                    help='outputfilename for the snr files')
parser.add_argument('--outputfilenameNuRadioReco', type=str, nargs='?', default=None,
                    help='outputfilename of NuRadioReco detector sim file')
parser.add_argument('--nchannels', type=int,
                    help='number of channels to phase', default=4)
args = parser.parse_args()

n_channels = args.nchannels

logger = module.setup_logger(level=logging.WARNING)

# initialize detector sim modules
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin(debug=False)
triggerSimulator = NuRadioReco.modules.phasedarray.triggerSimulator.triggerSimulator()
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
thresholdSimulator = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()

# 4 channel, 2x sampling, fft upsampling, 16 ns window
# 100 Hz -> 30.85
# 10 Hz -> 35.67
# 1 Hz -> 41.35

# 8 channel, 4x sampling, fft upsampling, 16 ns window
# 100 Hz -> 62.15
# 10 Hz -> 69.06
# 1 Hz -> 75.75

main_low_angle = np.deg2rad(-59.55)
main_high_angle = np.deg2rad(59.55)

N = 11
SNRs = (np.linspace(0.1, 4.0, N))
SNRtriggered = np.zeros(N)

channels = []

min_freq = 0.0 * units.MHz
max_freq = 250.0 * units.MHz
fff = np.linspace(min_freq, max_freq, 10000)
filt1_highres = channelBandPassFilter.get_filter(fff, 0, 0, None, passband=[0, 220 * units.MHz], filter_type="cheby1", order=7, rp=.1)
filt2_highres = channelBandPassFilter.get_filter(fff, 0, 0, None, passband=[96 * units.MHz, 100 * units.GHz], filter_type="cheby1", order=4, rp=.1)
filt_highres = filt1_highres * filt2_highres
bandwidth = np.trapz(np.abs(filt_highres) ** 2, fff)
Vrms_ratio = np.sqrt(bandwidth / (max_freq - min_freq))

Vrms = 1

new_sampling_rate = 500.0 * units.MHz

if(n_channels == 4):
    upsampling_factor = 2
    window = int(16 * units.ns * new_sampling_rate * upsampling_factor)
    step = int(8 * units.ns * new_sampling_rate * upsampling_factor)
    phasing_angles = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 11))
    channels = np.arange(4, 8)
    threshold = 30.85 * np.power(Vrms, 2.0)
elif(n_channels == 8):
    upsampling_factor = 4
    window = int(16 * units.ns * new_sampling_rate * upsampling_factor)
    step = int(8 * units.ns * new_sampling_rate * upsampling_factor)
    phasing_angles = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 21))
    channels = np.arange(8)
    threshold = 62.15 * np.power(Vrms, 2.0)
else:
    print("wrong n_channels!")
    exit()


def count_events():
    count_events.events += 1


count_events.events = 0


class mySimulation(simulation.simulation):

    def _detector_simulation_filter_amp(self, evt, station, det):

        channelBandPassFilter.run(evt, station, det, passband=[0.0 * units.MHz, 220.0 * units.MHz],
                                  filter_type='cheby1', order=9, rp=.1)
        channelBandPassFilter.run(evt, station, det, passband=[96.0 * units.MHz, 100.0 * units.GHz],
                                  filter_type='cheby1', order=4, rp=.1)

    def _detector_simulation_trigger(self, evt, station, det):
        # Start detector simulation

        orig_traces = {}
        for channel in station.iter_channels():
            orig_traces[channel.get_id()] = channel.get_trace()

            # If there is an empty trace, leave
            if(np.sum(channel.get_trace()) == 0):
                return

        filtered_signal_traces = {}
        for channel in station.iter_channels(use_channels=channels):
            trace = np.array(channel.get_trace())
            channel.set_trace(trace, new_sampling_rate)
            filtered_signal_traces[channel.get_id()] = trace

        # Since there are often two traces within the same thing, gotta be careful
        Vpps = []
        for channel in station.iter_channels(use_channels=channels):
            trace = np.array(channel.get_trace())
            Vpps += [np.max(trace) - np.min(trace)]
        Vpps = np.array(Vpps)

        # loop over all channels (i.e. antennas) of the station
        for channel in station.iter_channels(use_channels=channels):
            trace = np.zeros(len(filtered_signal_traces[channel.get_id()][:]))
            channel.set_trace(trace, sampling_rate=new_sampling_rate)

        # Adding noise AFTER the SNR calculation
        channelGenericNoiseAdder.run(evt, station, det, amplitude=Vrms / Vrms_ratio,
                                     min_freq=min_freq, max_freq=max_freq, type='rayleigh')

        # bandpass filter trace, the upper bound is higher then the sampling rate which makes it just a highpass filter
        channelBandPassFilter.run(evt, station, det, passband=[0 * units.MHz, 220.0 * units.MHz],
                                  filter_type='cheby1', order=7, rp=.1)
        channelBandPassFilter.run(evt, station, det, passband=[96.0 * units.MHz, 100.0 * units.GHz],
                                  filter_type='cheby1', order=4, rp=.1)

        filtered_noise_traces = {}
        for channel in station.iter_channels(use_channels=channels):
            filtered_noise_traces[channel.get_id()] = channel.get_trace()

        factor = 1.0 / (np.mean(Vpps) / (2.0 * Vrms))
        mult_factors = factor * SNRs

        if(np.mean(Vpps) == 0.0):
            return

        for factor, iSNR in zip(mult_factors, range(len(mult_factors))):

            for channel in station.iter_channels(use_channels=channels):
                trace = copy.deepcopy(filtered_signal_traces[channel.get_id()][:]) * factor
                noise = copy.deepcopy(filtered_noise_traces[channel.get_id()])
                channel.set_trace(trace + noise, sampling_rate=new_sampling_rate)

            has_triggered = triggerSimulator.run(evt, station, det,
                                                 Vrms=Vrms,
                                                 threshold=threshold,
                                                 triggered_channels=channels,
                                                 phasing_angles=phasing_angles,
                                                 ref_index=1.75,
                                                 trigger_name='primary_phasing',
                                                 trigger_adc=False,
                                                 adc_output='voltage',
                                                 trigger_filter=None,
                                                 upsampling_factor=upsampling_factor,
                                                 window=window,
                                                 step=step)

            if(has_triggered):
                print('Trigger for SNR', SNRs[iSNR])
                SNRtriggered[iSNR] += 1

        count_events()
        print(count_events.events)
        print(SNRtriggered)

        if(count_events.events % 10 == 0):
            output = {'total_events': count_events.events, 'SNRs': list(SNRs), 'triggered': list(SNRtriggered)}
            outputfile = args.outputSNR
            with open(outputfile, 'w+') as fout:
                json.dump(output, fout, sort_keys=True, indent=4)


sim = mySimulation(
    inputfilename=args.inputfilename,
    outputfilename=args.outputfilename,
    detectorfile=args.detectordescription,
    outputfilenameNuRadioReco=args.outputfilenameNuRadioReco,
    config_file=args.config,
    default_detector_station=1
)
sim.run()

print("Total events", count_events.events)
print("SNRs: ", SNRs)
print("Triggered: ", SNRtriggered)

output = {'total_events': count_events.events, 'SNRs': list(SNRs), 'triggered': list(SNRtriggered)}

outputfile = args.outputSNR
with open(outputfile, 'w+') as fout:
    json.dump(output, fout, sort_keys=True, indent=4)
