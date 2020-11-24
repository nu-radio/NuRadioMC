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
# 100 Hz -> 1.77
# 10 Hz -> 1.98
# 1 Hz -> 2.20

# 8 channel, 4x sampling, fft upsampling, 16 ns window
#  100 Hz -> 1.83
#  10 Hz -> 2.05
#  1 Hz -> 2.26

# 4 channels, 2x sampling, 16 ns window, linear up
#  100 Hz -> 1.26
#  10 Hz -> 1.43
#  1 Hz -> 1.60

# 8 channels, 4x sampling, 16 ns window, linear up
#  100 Hz -> 1.19
#  10 Hz -> 1.34
#  1 Hz -> 1.50

n_channels = 8
main_low_angle = np.deg2rad(-59.55)
main_high_angle = np.deg2rad(59.55)

N = 31
SNRs = (np.linspace(0.1, 4.0, N))
SNRtriggered = np.zeros(N)

channels = []

min_freq = 0.0 * units.MHz
max_freq = 250.0 * units.MHz
fff = np.linspace(min_freq, max_freq, 10000)
filt1_highres = channelBandPassFilter.get_filter(fff, 0, 0, None, passband=[0, 240 * units.MHz], filter_type="cheby1", order=9, rp=.1)
filt2_highres = channelBandPassFilter.get_filter(fff, 0, 0, None, passband=[80 * units.MHz, 230 * units.MHz], filter_type="cheby1", order=4, rp=.1)
filt_highres = filt1_highres * filt2_highres
bandwidth = np.trapz(np.abs(filt_highres) ** 2, fff)
Vrms_ratio = np.sqrt(bandwidth / (max_freq - min_freq))

new_sampling_rate = 500.0 * units.MHz

if(n_channels == 4):
    upsampling_factor = 2
    window = int(16 * units.ns * new_sampling_rate * upsampling_factor)
    step = int(8 * units.ns * new_sampling_rate * upsampling_factor)
    phasing_angles = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 11))
    channels = np.arange(4, 8)
    threshold = 1.77 * window
    phase = True
elif(n_channels == 8):
    upsampling_factor = 4
    window = int(16 * units.ns * new_sampling_rate * upsampling_factor)
    step = int(8 * units.ns * new_sampling_rate * upsampling_factor)
    phasing_angles = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 21))
    channels = None
    threshold = 1.83 * window
    phase = True
elif(n_channels == 1):
    upsampling_factor = 1
    window = int(16 * units.ns * new_sampling_rate * upsampling_factor)
    step = int(8 * units.ns * new_sampling_rate * upsampling_factor)
    phasing_angles = []
    channels = [4]
    threshold = 2.5
    phase = False
else:
    print("wrong n_channels!")
    exit()


def count_events():
    count_events.events += 1


count_events.events = 0


class mySimulation(simulation.simulation):

    def _detector_simulation_filter_amp(self, evt, station, det):
        channelBandPassFilter.run(evt, station, det, passband=[0.0 * units.MHz, 240.0 * units.MHz],
                                  filter_type='cheby1', order=9, rp=.1)
        channelBandPassFilter.run(evt, station, det, passband=[80.0 * units.MHz, 230.0 * units.MHz],
                                  filter_type='cheby1', order=4, rp=.1)

    def _detector_simulation_part2(self):
        # Start detector simulation

        # Convolve efield with antenna pattern
        efieldToVoltageConverter.run(self._evt, self._station, self._det)

        # Downsample trace back to detector sampling rate
        channelResampler.run(self._evt, self._station, self._det, sampling_rate=new_sampling_rate)

        # Filter signals
        channelBandPassFilter.run(self._evt, self._station, self._det, passband=[0.0 * units.MHz, 240.0 * units.MHz],
                                  filter_type='cheby1', order=9, rp=.1)
        channelBandPassFilter.run(self._evt, self._station, self._det, passband=[80.0 * units.MHz, 230.0 * units.MHz],
                                  filter_type='cheby1', order=4, rp=.1)

        filtered_signal_traces = {}
        for channel in self._station.iter_channels():
            trace = np.array(channel.get_trace())
            filtered_signal_traces[channel.get_id()] = trace

        # Since there are often two traces within the same thing, gotta be careful
        Vpps = np.zeros(self._station.get_number_of_channels())
        for iChan, channel in enumerate(self._station.iter_channels()):
            trace = np.array(channel.get_trace())
            Vpps[iChan] = np.max(trace) - np.min(trace)

        factor = 1.0 / (np.mean(Vpps) / 2.0)
        mult_factors = factor * SNRs

        has_triggered = True
        while(has_triggered):  # search for noise traces that don't set off a trigger

            # loop over all channels (i.e. antennas) of the station
            for channel in self._station.iter_channels():
                trace = np.zeros(len(filtered_signal_traces[channel.get_id()][:]))
                channel.set_trace(trace, sampling_rate=new_sampling_rate)

            # Adding noise AFTER the SNR calculation
            channelGenericNoiseAdder.run(self._evt, self._station, self._det, amplitude=1.0 / Vrms_ratio,
                                         min_freq=min_freq, max_freq=max_freq, type='rayleigh')

            # bandpass filter trace, the upper bound is higher then the sampling rate which makes it just a highpass filter
            channelBandPassFilter.run(self._evt, self._station, self._det, passband=[0 * units.MHz, 240.0 * units.MHz],
                                      filter_type='cheby1', order=9, rp=.1)
            channelBandPassFilter.run(self._evt, self._station, self._det, passband=[80.0 * units.MHz, 230.0 * units.MHz],
                                      filter_type='cheby1', order=4, rp=.1)

            if(phase):
                has_triggered = triggerSimulator.run(self._evt, self._station, self._det,
                                                     Vrms=1.0,
                                                     threshold=threshold,
                                                     triggered_channels=channels,
                                                     phasing_angles=phasing_angles,
                                                     ref_index=1.75,
                                                     trigger_name='primary_phasing',
                                                     trigger_adc=False,  # Don't have a seperate ADC for the trigger
                                                     clock_offset=np.random.uniform(0.0, 2.0),
                                                     adc_output='voltage',  # output in volts
                                                     trigger_filter=None,
                                                     upsampling_factor=upsampling_factor,
                                                     window=window,
                                                     step=step)
            else:
                original_traces_ = self._station.get_channel(4).get_trace()

                squared_mean, num_frames = triggerSimulator.power_sum(coh_sum=original_traces_,
                                                                      window=window,
                                                                      step=step,
                                                                      adc_output='voltage')

                squared_mean_threshold = np.power(threshold, 2.0)

                has_triggered = (True in (squared_mean > squared_mean_threshold))

            if(has_triggered):
                print('Trigger on noise... ')

        filtered_noise_traces = {}
        for channel in self._station.iter_channels():
            filtered_noise_traces[channel.get_id()] = channel.get_trace()

        for factor, iSNR in zip(mult_factors, range(len(mult_factors))):

            for channel in self._station.iter_channels():
                trace = copy.deepcopy(filtered_signal_traces[channel.get_id()][:]) * factor
                noise = filtered_noise_traces[channel.get_id()]
                channel.set_trace(trace + noise, sampling_rate=new_sampling_rate)

            if(phase):
                has_triggered = triggerSimulator.run(self._evt, self._station, self._det,
                                                     Vrms=1.0,
                                                     threshold=threshold,
                                                     triggered_channels=channels,
                                                     phasing_angles=phasing_angles,
                                                     ref_index=1.75,
                                                     trigger_name='primary_phasing',
                                                     trigger_adc=False,
                                                     clock_offset=np.random.uniform(0.0, 2.0),
                                                     adc_output='voltage',
                                                     trigger_filter=None,
                                                     upsampling_factor=upsampling_factor,
                                                     window=window,
                                                     step=step)
            else:
                original_traces_ = self._station.get_channel(4).get_trace()

                squared_mean, num_frames = triggerSimulator.power_sum(coh_sum=original_traces_,
                                                                      window=window,
                                                                      step=step,
                                                                      adc_output='voltage')

                squared_mean_threshold = np.power(threshold, 2.0)

                has_triggered = (True in (squared_mean > squared_mean_threshold))

            if(has_triggered):
                print('Trigger for SNR', SNRs[iSNR])
                SNRtriggered[iSNR] += 1

        count_events()
        print(count_events.events)
        print(SNRtriggered)

        if(count_events.events % 10 == 0):
            # plt.show()
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
