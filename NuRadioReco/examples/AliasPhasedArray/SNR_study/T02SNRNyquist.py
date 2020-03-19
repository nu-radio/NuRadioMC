"""
This file calculates the SNR curves for a phased array containing a trigger ADC
and using alias as a trigger mechanism, i.e., using the higher Nyquist zones.

Neutrinos specified by the input file should interact probably within the region
where triggers are likely (see Fig. 6 from the lepton propagation paper) and
come from the most likely arrival directions (see Fig. 7 from the same paper),
so that the simulation is faster. See T01_generate_events_simple.py.

The phased array configuration is the same used for the RNO example.

The usage of this file is:

python T02SNRNyquist.py input_neutrino_file.hdf5 phased_array_file.json
config.yaml output_NuRadioMC_file.hdf5 output_SNR_file.hdf5 output_NuRadioReco_file.nur(optional)

The Nyquist zone and the upsampling factors can be changed in this file. The
following configurations are the most interesting:

            ADC freq (GHz)        Upsampling factor     Nyquist zone
Config 1        0.5                     2                   1
Config 2        0.25                    4                   2
Config 3        0.25                    4                   3

This folder contains two files, one with ADCs of a sampling frequency of 0.25
GHz, and another with 0.5 GHz. The refrence voltages have been chosen so that
the noise RMS uses 2 ADC bits, but they can be changed by the user.

The output_SNR_file.hdf5 contains the information on the SNR curves stored in
hdf5 format. The fields in this file are:
    - 'total_events', the total number of events that have been used
    - 'SNRs', a list with the SNR

The antenna positions can be changed in the detector position. The rest of the
properties  of the phased array can be changed in the current file - phasing angles,
triggering channels, bandpass filter and so on. Just be aware that tweaking
with these parameters will change the noise trigger rate, which should be
recalculated for having realistic results.

WARNING: this file needs NuRadioMC to be run.
"""

from __future__ import absolute_import, division, print_function
import argparse
# import detector simulation modules
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.phasedarray.triggerSimulator
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.utilities.diodeSimulator
from NuRadioReco.utilities.traceWindows import get_window_around_maximum
from NuRadioReco.utilities import units
from NuRadioMC.simulation import simulation
from NuRadioReco.modules import analogToDigitalConverter
from NuRadioReco.utilities.trace_utilities import butterworth_filter_trace
import numpy as np
import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("runstrawman")

# initialize detector sim modules
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin(debug=False)
triggerSimulator = NuRadioReco.modules.phasedarray.triggerSimulator.triggerSimulator()
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
thresholdSimulator = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()
ADC = NuRadioReco.modules.analogToDigitalConverter.analogToDigitalConverter()

upsampling_factor = 4
nyquist_zone = 2

main_low_angle = -50 * units.deg
main_high_angle = 50 * units.deg
phasing_angles = np.arcsin( np.linspace( np.sin(main_low_angle), np.sin(main_high_angle), 30) )

diode_passband = (None, 200*units.MHz)
diodeSimulator = NuRadioReco.utilities.diodeSimulator.diodeSimulator(diode_passband)

thresholds = {'try': 3e-5}

low_freq = 132 * units.MHz
high_freq = 700 * units.MHz

N = 51
SNRs = np.linspace(0.5,5,N)
SNRtriggered = np.ones(N) * -1
def count_events():
    count_events.events += 1
count_events.events = -1

class mySimulation(simulation.simulation):

    def _detector_simulation(self):
        # start detector simulation
        efieldToVoltageConverter.run(self._evt, self._station, self._det)  # convolve efield with antenna pattern
        # downsample trace to 5 ns
        new_sampling_rate = 5 * units.GHz
        channelResampler.run(self._evt, self._station, self._det, sampling_rate=new_sampling_rate)

        cut_times = get_window_around_maximum(self._station, diodeSimulator, ratio=0.01)

        # Calculating peak to peak voltage and the necessary factors for the SNR.
        Vpps = []
        for channel in self._station.iter_channels():

            times = np.array(channel.get_times())
            trace = np.array(channel.get_trace())

            trace = butterworth_filter_trace(trace, new_sampling_rate, (low_freq, high_freq))
            left_bin = np.argmin(np.abs(times-cut_times[0]))
            right_bin = np.argmin(np.abs(times-cut_times[1]))

            Vpp = np.max( trace[left_bin:right_bin] ) - np.min( trace[left_bin:right_bin] )
            Vpps.append(Vpp)

        factor = 1./(np.mean(Vpps)/2/self._Vrms)
        print(factor)
        mult_factors = factor * SNRs

        reject_event = False
        if True in mult_factors > 1.e10:
            reject_event = True

        # Copying original traces
        original_traces = {}

        for channel in self._station.iter_channels():

            trace = np.array( channel.get_trace() )
            channel_id = channel.get_id()
            original_traces[channel_id] = trace

        # Looping over the different SNRs to calculate the efficiencies
        for factor, iSNR in zip(mult_factors, range(len(mult_factors))):

            for channel in self._station.iter_channels():

                trace = original_traces[channel.get_id()][:] * factor
                channel.set_trace( trace, sampling_rate = new_sampling_rate )

            noise = True

            if noise:
                max_freq = 0.5 * new_sampling_rate
                norm = self._get_noise_normalization(self._station.get_id())  # assuming the same noise level for all stations
                Vrms = self._Vrms / (norm / (max_freq)) ** 0.5  # normalize noise level to the bandwidth its generated for
                channelGenericNoiseAdder.run(self._evt, self._station, self._det, amplitude=Vrms, min_freq=0 * units.MHz,
                                             max_freq=max_freq, type='rayleigh')

            # bandpass filter trace, the upper bound is higher then the sampling rate which makes it just a highpass filter
            channelBandPassFilter.run(self._evt, self._station, self._det, passband=[low_freq, 1150 * units.MHz],
                                      filter_type='butter', order=8)
            channelBandPassFilter.run(self._evt, self._station, self._det, passband=[0, high_freq],
                                      filter_type='butter', order=10)

            # Running the phased array trigger with ADC, Nyquist zones and upsampling incorporated
            trig = triggerSimulator.run(self._evt, self._station, self._det,
                                 threshold=thresholds['try'], # see phased trigger module for explanation
                                 triggered_channels=None,  # run trigger on all channels
                                 trigger_name='primary_phasing', # the name of the trigger
                                 phasing_angles=phasing_angles,
                                 secondary_phasing_angles=None,
                                 coupled=False,
                                 ref_index=1.55,
                                 cut_times=cut_times,
                                 trigger_adc=True,
                                 upsampling_factor=upsampling_factor,
                                 nyquist_zone=nyquist_zone,
                                 bandwidth_edge=20*units.MHz)

            if trig:
                ADC.run(self._evt, self._station, self._det)

            if (trig and not reject_event):
                print('Trigger for SNR', SNRs[iSNR])
                SNRtriggered[iSNR] += 1

        if not reject_event:
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

sim = mySimulation(inputfilename=args.inputfilename,
                            outputfilename=args.outputfilename,
                            detectorfile=args.detectordescription,
                            outputfilenameNuRadioReco=args.outputfilenameNuRadioReco,
                            config_file=args.config)
sim.run()

print("Total events", count_events.events)
print("SNRs: ", SNRs)
print("Triggered: ", SNRtriggered)

output = {}
output['total_events'] = count_events.events
output['SNRs'] = list( SNRs )
output['triggered'] = list( SNRtriggered )

outputfile = args.outputSNR
with open(outputfile, 'w+') as fout:

    json.dump(output, fout, sort_keys=True, indent=4)
