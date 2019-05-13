from __future__ import absolute_import, division, print_function
import argparse
# import detector simulation modules
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.triggerSimulator
import NuRadioReco.modules.phasedarray.triggerSimulator
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelGenericNoiseAdder
from NuRadioReco.utilities import units
from NuRadioMC.simulation import simulation2 as simulation
import json
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("runstrawman")

# initialize detector sim modules
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin(debug=False, time_resolution=1*units.ns)
triggerSimulator = NuRadioReco.modules.phasedarray.triggerSimulator.triggerSimulator()
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
thresholdSimulator = NuRadioReco.modules.triggerSimulator.triggerSimulator()

N = 51
SNRs = np.linspace(0.5,5,N)
SNRtriggered = np.zeros(N)
def count_events():
    count_events.events += 1
count_events.events = 0

class mySimulation(simulation.simulation):

    def _detector_simulation(self):
        # start detector simulation
        efieldToVoltageConverter.run(self._evt, self._station, self._det)  # convolve efield with antenna pattern
        # downsample trace back to detector sampling rate
        new_sampling_rate = 1.5 * units.GHz
        channelResampler.run(self._evt, self._station, self._det, sampling_rate=new_sampling_rate)

        reject_event = False
        threshold_cut = True
        # Forcing a threshold cut BEFORE adding noise for limiting the noise-induced triggers
        if threshold_cut:

            thresholdSimulator.run(self._evt, self._station, self._det,
                                 threshold=1 * self._Vrms,
                                 triggered_channels=None,  # run trigger on all channels
                                 number_concidences=1,
                                 trigger_name='simple_threshold')

            if not self._station.get_trigger('simple_threshold').has_triggered():
                reject_event = True
                print("You shall not pass (the threshold)")

        # Copying the original traces. Not really needed
        original_traces = {}

        for channel in self._station.iter_channels():  # loop over all channels (i.e. antennas) of the station

            trace = np.array( channel.get_trace() )
            channel_id = channel.get_id()
            original_traces[channel_id] = trace

        # Finding bins for each channel so as to calculate the peak to peak voltage
        ext_bins = {}

        for channel in self._station.iter_channels():  # loop over all channels (i.e. antennas) of the station

            trace = np.array( channel.get_trace() )
            channel_id = channel.get_id()
            max_bin = np.argwhere( trace == np.max(trace) )[0,0]
            min_bin = np.argwhere( trace == np.min(trace) )[0,0]
            if ( max_bin < min_bin ):
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
            trace = np.array( channel.get_trace() )
            try:
                Vpp = np.max(trace[left_bin:right_bin])-np.min(trace[left_bin:right_bin])
            except:
                Vpp = 0
                #reject_event = True
            Vpps.append(Vpp)

        factor = 1./(np.mean(Vpps)/2/self._Vrms)
        #factor = 0
        mult_factors = factor * SNRs
        print(factor)

        if True in mult_factors > 1.e10:
            reject_event = True

        for factor, iSNR in zip(mult_factors, range(len(mult_factors))):

            for channel in self._station.iter_channels():  # loop over all channels (i.e. antennas) of the station

                trace = original_traces[channel.get_id()][:] * factor
                channel.set_trace( trace, sampling_rate = new_sampling_rate )

            min_freq = 100 * units.MHz
            max_freq = 750 * units.MHz
            Vrms_ratio = ((max_freq-min_freq) / self._cfg['trigger']['bandwidth'])**2
            # Adding noise AFTER the SNR calculation
            channelGenericNoiseAdder.run(self._evt, self._station, self._det, amplitude=self._Vrms*Vrms_ratio,
                                         min_freq=min_freq,
                                         max_freq=max_freq,
                                         type='rayleigh')

            # bandpass filter trace, the upper bound is higher then the sampling rate which makes it just a highpass filter
            channelBandPassFilter.run(self._evt, self._station, self._det, passband=[130 * units.MHz, 1000 * units.GHz],
                                      filter_type='butter', order=2)
            channelBandPassFilter.run(self._evt, self._station, self._det, passband=[0, 750 * units.MHz],
                                      filter_type='butter', order=10)

            # Phased array with ARA-like power trigger
            has_triggered = triggerSimulator.run(self._evt, self._station, self._det,
                                 threshold= 2.5 * self._Vrms,
                                 triggered_channels=None,  # run trigger on all channels
                                 trigger_name='primary_and_secondary_phasing',
                                 set_not_triggered=(not self._station.has_triggered("simple_threshold")))

            if (has_triggered and not reject_event):
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

sim = mySimulation(eventlist=args.inputfilename,
                            outputfilename=args.outputfilename,
                            detectorfile=args.detectordescription,
                            station_id=101,
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
