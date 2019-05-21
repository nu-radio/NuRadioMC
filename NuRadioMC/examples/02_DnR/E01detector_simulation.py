from __future__ import absolute_import, division, print_function
import argparse
from six import iteritems
# import detector simulation modules
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.ARIANNA.triggerSimulator
import NuRadioReco.modules.triggerSimulator
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.custom.deltaT.calculateAmplitudePerRaySolution
from NuRadioReco.utilities import units
from NuRadioMC.simulation import simulation2 as simulation
from NuRadioReco.framework.parameters import electricFieldParameters as efp
import logging
import numpy as np
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("runDeltaTStudy")

# initialize detector sim modules
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin(pre_pulse_time=0 * units.ns, post_pulse_time=0 * units.ns)
calculateAmplitudePerRaySolution = NuRadioReco.modules.custom.deltaT.calculateAmplitudePerRaySolution.calculateAmplitudePerRaySolution()
triggerSimulator = NuRadioReco.modules.triggerSimulator.triggerSimulator()
triggerSimulatorARIANNA = NuRadioReco.modules.ARIANNA.triggerSimulator.triggerSimulator()
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()


class mySimulation(simulation.simulation):


    def _detector_simulation(self):
        if(bool(self._cfg['signal']['zerosignal'])):
            self._increase_signal(None, 0)

        calculateAmplitudePerRaySolution.run(self._evt, self._station, self._det)
        # save the amplitudes to output hdf5 file
        # save amplitudes per ray tracing solution to hdf5 data output
        sg = self._mout_groups[self._station_id]
        n_antennas = self._det.get_number_of_channels(self._station_id)
        if('max_amp_ray_solution' not in sg):
            sg['max_amp_ray_solution'] = np.zeros((self._n_events, n_antennas, 2)) * np.nan
        ch_counter = np.zeros(n_antennas, dtype=np.int)
        for efield in self._station.get_sim_station().get_electric_fields():
            for channel_id, maximum in iteritems(efield[efp.max_amp_antenna]):
                sg['max_amp_ray_solution'][self._iE, channel_id, ch_counter[channel_id]] = maximum
                ch_counter[channel_id] += 1 
        
        # start detector simulation
        efieldToVoltageConverter.run(self._evt, self._station, self._det)  # convolve efield with antenna pattern
        
        # downsample trace to internal simulation sampling rate (the efieldToVoltageConverter upsamples the trace to
        # 20 GHz by default to achive a good time resolution when the two signals from the two signal paths are added) 
        channelResampler.run(self._evt, self._station, self._det, sampling_rate=1. / self._dt)
        
        if bool(self._cfg['noise']):
            Vrms = self._Vrms / (self._bandwidth /( 2.5 * units.GHz))** 0.5  # normalize noise level to the bandwidth its generated for
            channelGenericNoiseAdder.run(self._evt, self._station, self._det, amplitude=Vrms, min_freq=0 * units.MHz,
                                         max_freq=2.5 * units.GHz, type='rayleigh')

        # bandpass filter trace, the upper bound is higher then the sampling rate which makes it just a highpass filter
        channelBandPassFilter.run(self._evt, self._station, self._det, passband=[80 * units.MHz, 1000 * units.GHz],
                                  filter_type='butter', order=2)
        channelBandPassFilter.run(self._evt, self._station, self._det, passband=[0, 500 * units.MHz],
                                  filter_type='butter', order=10)
        triggerSimulator.run(self._evt, self._station, self._det,
                             threshold=2 * self._Vrms,
                             triggered_channels=None,
                             number_concidences=1,
                             trigger_name='pre_trigger_2sigma')
        
        # downsample trace back to detector sampling rate
        channelResampler.run(self._evt, self._station, self._det, sampling_rate=self._sampling_rate_detector)


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
