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
defines de bandwidth for the noise RMS calculation. The properties of the phased
array can be changed in the current file - phasing angles, triggering channels,
bandpass filter and so on.

WARNING: this file needs NuRadioMC to be run.
"""

from __future__ import absolute_import, division, print_function
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import logging

import sys
sys.path.append('/home/danielsmith/icecube_gen2/NuRadioReco')

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

N = 51
SNRs = (np.linspace(0.5, 5, N))
SNRtriggered = np.zeros(N)

def count_events():
    count_events.events += 1

count_events.events = 0

class mySimulation(simulation.simulation):

    def _detector_simulation_filter_amp(self, evt, station, det):
        channelBandPassFilter.run(
            evt,
            station,
            det,
            passband=[130 * units.MHz, 1000 * units.GHz],
            filter_type='butter',
            order=2
        )
        channelBandPassFilter.run(
            evt,
            station,
            det,
            passband=[0, 750 * units.MHz],
            filter_type='butter',
            order=10
        )

    def _detector_simulation_part2(self):
        # start detector simulation
        efieldToVoltageConverter.run(self._evt, self._station, self._det)  # convolve efield with antenna pattern

        # downsample trace back to detector sampling rate
        new_sampling_rate = 1.5 * units.GHz
        channelResampler.run(self._evt, self._station, self._det, sampling_rate=new_sampling_rate)

        # Copying the original traces. Not really needed
        original_traces = {}

        for channel in self._station.iter_channels():  # loop over all channels (i.e. antennas) of the station

            trace = np.array(channel.get_trace())
            channel_id = channel.get_id()
            original_traces[channel_id] = trace

        # Finding bins for each channel so as to calculate the peak to peak voltage
        ext_bins = {}

        for channel in self._station.iter_channels():  # loop over all channels (i.e. antennas) of the station

            trace = np.array(channel.get_trace())
            channel_id = channel.get_id()
            max_bin = np.argwhere(trace == np.max(trace))[0, 0]
            min_bin = np.argwhere(trace == np.min(trace))[0, 0]
            if (max_bin < min_bin):
                left_bin = max_bin - 20
                right_bin = min_bin + 20
            else:
                left_bin = min_bin - 20
                right_bin = max_bin + 20
            ext_bins[channel_id] = (left_bin, right_bin)

        # Calculating peak to peak voltage and the necessary factors for the SNR.
        Vpps = []
        for channel in self._station.iter_channels():  # loop over all channels (i.e. antennas) of the station

            left_bin, right_bin = ext_bins[channel.get_id()]
            trace = np.array(channel.get_trace())
            try:
                Vpp = np.max(trace[left_bin:right_bin]) - np.min(trace[left_bin:right_bin])
            except:
                Vpp = 0
                # reject_event = True
            Vpps.append(Vpp)

        factor = 1. / (np.mean(Vpps) / 2 / self._Vrms)
        # factor = 0
        mult_factors = factor * SNRs

        for factor, iSNR in zip(mult_factors, range(len(mult_factors))):

            for channel in self._station.iter_channels():  # loop over all channels (i.e. antennas) of the station

                trace = original_traces[channel.get_id()][:] * factor
                channel.set_trace(trace, sampling_rate=new_sampling_rate)

            min_freq = 100 * units.MHz
            max_freq = 750 * units.MHz
            Vrms_ratio = ((max_freq - min_freq) / self._cfg['trigger']['bandwidth']) ** 2

            # Adding noise AFTER the SNR calculation
            channelGenericNoiseAdder.run(self._evt, self._station, self._det, amplitude = self._Vrms * Vrms_ratio,
                                         min_freq=min_freq,
                                         max_freq=max_freq,
                                         type='rayleigh')

            # bandpass filter trace, the upper bound is higher then the sampling rate which makes it just a highpass filter
            channelBandPassFilter.run(self._evt, self._station, self._det, passband=[130 * units.MHz, 1000 * units.GHz],
                                      filter_type='butter', order=2)
            channelBandPassFilter.run(self._evt, self._station, self._det, passband=[0, 750 * units.MHz],
                                      filter_type='butter', order=10)

            voltage_to_adc = (2**(5-1) - 1) / self._Vrms

            # Phased array with ARA-like power trigger
            has_triggered = triggerSimulator.run(self._evt,
                                                 self._station,
                                                 self._det,
                                                 Vrms = self._Vrms,
                                                 threshold = 1.5 * self._Vrms * voltage_to_adc,
                                                 triggered_channels=None,  # run trigger on all channels
                                                 trigger_name='primary_phasing',
                                                 ref_index=1.78)

            if(has_triggered):
                print('Trigger for SNR', SNRs[iSNR])
                SNRtriggered[iSNR] += 1

        count_events()
        print(count_events.events)
        print(SNRtriggered)


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
