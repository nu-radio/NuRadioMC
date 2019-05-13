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
import numpy as np
import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("runstrawman")

# initialize detector sim modules
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin(debug=False, time_resolution=1*units.ns)
triggerSimulator = NuRadioReco.modules.phasedarray.triggerSimulator.triggerSimulator()
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
thresholdSimulator = NuRadioReco.modules.triggerSimulator.triggerSimulator()

class mySimulation(simulation.simulation):

    def _detector_simulation(self):
        # start detector simulation
        efieldToVoltageConverter.run(self._evt, self._station, self._det)  # convolve efield with antenna pattern
        # downsample trace to 3 ns
        new_sampling_rate = 1.5 * units.GHz
        channelResampler.run(self._evt, self._station, self._det, sampling_rate=new_sampling_rate)

        threshold_cut = True
        # Forcing a threshold cut BEFORE adding noise for limiting the noise-induced triggers
        if threshold_cut:

            thresholdSimulator.run(self._evt, self._station, self._det,
                                 threshold=1 * self._Vrms,
                                 triggered_channels=None,  # run trigger on all channels
                                 number_concidences=1,
                                 trigger_name='simple_threshold')

        # Bool for checking the noise triggering rate
        check_only_noise = False
        if check_only_noise:

            for channel in self._station.iter_channels():  # loop over all channels (i.e. antennas) of the station

                trace = channel.get_trace() * 0
                channel.set_trace(trace, sampling_rate = new_sampling_rate)

        noise = True

        if noise:
            min_noise_freq = 100 * units.MHz
            max_noise_freq = 750 * units.MHz
            Vrms_ratio = ((max_noise_freq-min_noise_freq) / self._cfg['trigger']['bandwidth'])**2
            channelGenericNoiseAdder.run(self._evt, self._station, self._det, amplitude=self._Vrms,
                                         min_freq=min_noise_freq,
                                         max_freq=max_noise_freq,
                                         type='rayleigh')
        # bandpass filter trace, the upper bound is higher then the sampling rate which makes it just a highpass filter
        channelBandPassFilter.run(self._evt, self._station, self._det, passband=[130 * units.MHz, 1000 * units.GHz],
                                  filter_type='butter', order=2)
        channelBandPassFilter.run(self._evt, self._station, self._det, passband=[0, 750 * units.MHz],
                                  filter_type='butter', order=10)

        # first run a simple threshold trigger
        triggerSimulator.run(self._evt, self._station, self._det,
                             threshold=2.5 * self._Vrms, # see phased trigger module for explanation
                             triggered_channels=None,  # run trigger on all channels
                             trigger_name='primary_and_secondary_phasing', # the name of the trigger
                             set_not_triggered=(not self._station.has_triggered("simple_threshold")),
                             only_primary=False,
                             coupled=True)


parser = argparse.ArgumentParser(description='Run NuRadioMC simulation')
parser.add_argument('inputfilename', type=str,
                    help='path to NuRadioMC input event list')
parser.add_argument('detectordescription', type=str,
                    help='path to file containing the detector description')
parser.add_argument('config', type=str,
                    help='NuRadioMC yaml config file')
parser.add_argument('outputfilename', type=str,
                    help='hdf5 output filename')
parser.add_argument('outputfilenameNuRadioReco', type=str, nargs='?', default=None,
                    help='outputfilename of NuRadioReco detector sim file')
args = parser.parse_args()

sim = mySimulation(eventlist=args.inputfilename,
                            outputfilename=args.outputfilename,
                            detectorfile=args.detectordescription,
                            outputfilenameNuRadioReco=args.outputfilenameNuRadioReco,
                            config_file=args.config)
sim.run()
