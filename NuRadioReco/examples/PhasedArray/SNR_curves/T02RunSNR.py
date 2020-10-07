"""
This file calculates the SNR curves for a given phased array configuration.
Neutrinos specified by the input file should come from a fixed arrival direction
if the SNR curve for that particular direction is required. It is convenient
that the array sees the neutrino at the Cherenkov angle, so that the signals
are clearer. The signals are then rescaled to have the appropriate SNR and
the phased trigger is run to decide whether the event is selected.

The phased array configuration
in this file is inspired by the deployed ARA phased array: 1.5 GS/s, 8 antennas
at a depth of ~50 m, 15 phasing directions with primary and secondary beams.
In order to run, we need a detector file and a configuration file, included in
this folder. To run the code, type:

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
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import logging
import copy 

# import detector simulation modules
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.phasedarray.triggerSimulator
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelGenericNoiseAdder
from NuRadioReco.utilities import units
from NuRadioMC.simulation import simulation
from NuRadioReco.modules.base import module

logger = module.setup_logger(level=logging.WARNING)

# initialize detector sim modules
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin(debug=False)
triggerSimulator = NuRadioReco.modules.phasedarray.triggerSimulator.triggerSimulator()
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
thresholdSimulator = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()

#import sys
#sys.path.append('/home/danielsmith/icecube_gen2/NuRadioReco')
#sys.path.append('/home/danielsmith/icecube_gen2/NuRadioMC')
            
# 4 channel, 2x sampling
#  100 Hz -> 2.91
#  10 Hz -> 3.06
#  1 Hz -> 3.20

# Half window integration
#  100 Hz -> 3.66
#  10 Hz -> 3.88
#  1 Hz -> 4.09

# 8 channel, 4x sampling
# 100 Hz -> 4.25
# 10 Hz -> 4.56
# 1 Hz -> 4.88

# Half window integration
# 100 Hz -> 5.18
#  10 Hz -> 5.46
#  1 Hz -> 5.71

n_channels = 4
phase = True

main_low_angle = np.deg2rad(-59.55)
main_high_angle = np.deg2rad(59.55)
channels = []

if(n_channels == 4):
    upsampling_factor = 2
    phasing_angles = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 11))
    channels = np.arange(4, 8)
    #threshold = 2.91
    threshold = 3.66
elif(n_channels == 8):
    upsampling_factor = 4
    phasing_angles = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 21))
    channels = None
    #threshold = 4.25
    threshold = 5.18
else:
    print("wrong n_channels!")
    exit()

N = 21
SNRs = (np.linspace(0.5, 4.0, N))
SNRtriggered = np.zeros(N)

def count_events():
    count_events.events += 1

count_events.events = 0

class mySimulation(simulation.simulation):

    def _detector_simulation_filter_amp(self, evt, station, det):
        channelBandPassFilter.run(evt, station, det, passband=[0, 500 * units.MHz],
                                  filter_type='butter', order=10)
        pass

    def _detector_simulation_part2(self):

        # start detector simulation
        efieldToVoltageConverter.run(self._evt, self._station, self._det)  # convolve efield with antenna pattern

        for channel in self._station.iter_channels():  # loop over all channels (i.e. antennas) of the station
            trace = np.array(channel.get_trace())
            channel_id = channel.get_id()
            # Do some magic here to find direction ... 

        # downsample trace back to detector sampling rate
        new_sampling_rate = 500.0 * units.MHz
        channelResampler.run(self._evt, self._station, self._det, sampling_rate=new_sampling_rate)

        # Copying the original traces. Not really needed
        original_traces = {}

        for channel in self._station.iter_channels():  # loop over all channels (i.e. antennas) of the station
            trace = np.array(channel.get_trace())

            # one quality check, problem with some signals being zero from being the the shadow, so
            if(np.sum(trace) == 0):
                print("Skipping event due to zero trace in", channel.get_id())
                return 

            channel_id = channel.get_id()
            original_traces[channel_id] = trace#_
            channel.set_trace(trace, new_sampling_rate)

        # Calculating peak to peak voltage and the necessary factors for the SNR.
        dt = 1 / new_sampling_rate
        ff = np.fft.rfftfreq(len(channel.get_trace()), dt)
        filt1 = channelBandPassFilter.get_filter(ff, 0, 0, None, passband=[0, 240 * units.MHz], filter_type="cheby1", order=9, rp=.1)
        filt2 = channelBandPassFilter.get_filter(ff, 0, 0, None, passband=[80 * units.MHz, 230 * units.MHz], filter_type="cheby1", order=4, rp=.1)

        # Since there are often two traces within the same thing, gotta be careful
        Vpps = []
        maxes = []
        for iChan, channel in enumerate(self._station.iter_channels()):  # loop over all channels (i.e. antennas) of the station
            trace = np.fft.irfft(np.fft.rfft(np.array(channel.get_trace())) * filt2 * filt1)
            Vpps += [np.max(trace) - np.min(trace)]

        factor = 1. / (np.mean(Vpps) / (2.0))# * self._Vrms))
        mult_factors = factor * SNRs

        for factor, iSNR in zip(mult_factors, range(len(mult_factors))):

            for channel in self._station.iter_channels():  # loop over all channels (i.e. antennas) of the station
                trace = copy.deepcopy(original_traces[channel.get_id()][:]) * factor
                channel.set_trace(trace, sampling_rate=new_sampling_rate)

            min_freq = 0.0 * units.MHz
            max_freq = 250.0 * units.MHz
            bandwidth = 0.1732429316625746 
            Vrms_ratio = np.sqrt(bandwidth / (max_freq - min_freq))

            # Adding noise AFTER the SNR calculation
            # no adding noise, see what that does to the SNR
            '''
            channelGenericNoiseAdder.run(self._evt, self._station, self._det, amplitude = self._Vrms / Vrms_ratio,
                                         min_freq=min_freq,
                                         max_freq=max_freq,
                                         type='rayleigh')
            '''

            # bandpass filter trace, the upper bound is higher then the sampling rate which makes it just a highpass filter
            channelBandPassFilter.run(self._evt, self._station, self._det, passband=[0 * units.MHz, 240 * units.MHz],
                                      filter_type='cheby1', order=9, rp=.1)
            channelBandPassFilter.run(self._evt, self._station, self._det, passband=[80 * units.MHz, 230 * units.MHz],
                                      filter_type='cheby1', order=4, rp=.1)

            if(phase):
                has_triggered = triggerSimulator.run(self._evt, self._station, self._det,
                                                     Vrms = 1.0, #self._Vrms,
                                                     threshold = threshold,
                                                     triggered_channels=channels,
                                                     phasing_angles=phasing_angles, 
                                                     ref_index = 1.75, 
                                                     trigger_name='primary_phasing',  # the name of the trigger
                                                     trigger_adc=False, # Don't have a seperate ADC for the trigger
                                                     adc_output='voltage', # output in volts
                                                     nyquist_zone=None, # first nyquist zone
                                                     bandwidth_edge=20 * units.MHz,                             
                                                     upsampling_factor=upsampling_factor,
                                                     window=int(32 * dt / 2 / upsampling_factor), 
                                                     step = int(16 * dt / 2 / upsampling_factor))
            else:
                thresholdSimulator.run(self._evt, self._station, self._det,
                                       threshold = 2.0, # * self._Vrms,
                                       triggered_channels=[7],  # run trigger on all channels
                                       number_concidences=1,
                                       trigger_name='simple_threshold')

                has_triggered = self._station.get_trigger('simple_threshold').has_triggered()

            if(has_triggered):
                print('Trigger for SNR', SNRs[iSNR])
                SNRtriggered[iSNR] += 1
        '''
            for channel in self._station.iter_channels():  # loop over all channels (i.e. antennas) of the station
                if(channel.get_id() != 4):
                    continue
                trace = copy.deepcopy(original_traces[channel.get_id()][:]) * factor
                colors = ['red', 'blue', 'green', 'orange', 'grey', 'black', 'pink', 'red', 'blue', 'green']
                plt.plot(trace, color=colors[count_events.events])
                print(maximum)
                plt.scatter(np.arange(len(trace))[maximum], trace[maximum])
        plt.show()
        '''

        count_events()
        print(count_events.events)
        print(SNRtriggered)

        if(count_events.events % 10 == 0):
            # Save every ten triggers
            output = {'total_events': count_events.events, 'SNRs': list(SNRs), 'triggered': list(SNRtriggered)}
            outputfile = args.outputSNR
            with open(outputfile, 'w+') as fout:
                json.dump(output, fout, sort_keys=True, indent=4)


parser = argparse.ArgumentParser(description='Run NuRadioMC simulation')
parser.add_argument('inputfilename', type=str,
                    help='path to NuRadioMC input event list')
parser.add_argument('detectordescription', type=str,
                    help='path to file containing the detector description')
parser.add_argument('config', type=str,
                    help='NuRadioMC yaml config file')
parser.add_argument('outputfilename', type=str,
                    help='hdf5 output filename')
parser.add_argument('outputSNR', type=str,
                    help='outputfilename for the snr files')
parser.add_argument('outputfilenameNuRadioReco', type=str, nargs='?', default=None,
                    help='outputfilename of NuRadioReco detector sim file')
args = parser.parse_args()

sim = mySimulation(
    inputfilename=args.inputfilename,
    outputfilename=args.outputfilename,
    detectorfile=args.detectordescription,
    outputfilenameNuRadioReco=args.outputfilenameNuRadioReco,
    config_file=args.config
)
sim.run()

print("Total events", count_events.events)
print("SNRs: ", SNRs)
print("Triggered: ", SNRtriggered)

output = {'total_events': count_events.events, 'SNRs': list(SNRs), 'triggered': list(SNRtriggered)}

outputfile = args.outputSNR
with open(outputfile, 'w+') as fout:
    json.dump(output, fout, sort_keys=True, indent=4)
