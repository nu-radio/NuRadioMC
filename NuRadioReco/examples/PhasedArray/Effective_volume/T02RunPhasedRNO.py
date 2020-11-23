"""
his file runs a phased array trigger simulation. The phased array configuration
in this file is similar to one of the proposed ideas for RNO: 3 GS/s, 8 antennas
at a depth of ~50 m, 30 primary phasing directions. In order to run, we need
a detector file and a configuration file, included in this folder. To run
the code, type:

python T02RunPhasedRNO.py input_neutrino_file.hdf5 4antennas_100m_1.5GHz.json
config_RNO.yaml output_NuRadioMC_file.hdf5 output_NuRadioReco_file.nur

The antenna positions can be changed in the detector position. The config file
defines de bandwidth for the noise RMS calculation. The properties of the phased
array can be changed in the current file - phasing angles, triggering channels,
bandpass filter and so on.

WARNING: this file needs NuRadioMC to be run.
"""
from __future__ import absolute_import, division, print_function

import argparse
import logging
import numpy as np
from NuRadioMC.simulation import simulation
import NuRadioReco
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.phasedarray.triggerSimulator
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelGenericNoiseAdder
from NuRadioReco.utilities import units
from NuRadioReco.modules.base import module

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

logger = module.setup_logger(level=logging.WARNING)

# initialize detector sim modules
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin(debug=False)

triggerSimulator = NuRadioReco.modules.phasedarray.triggerSimulator.triggerSimulator()
simpleThreshold = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()

channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()

main_low_angle = np.deg2rad(-59.54968597864437)
main_high_angle = np.deg2rad(59.54968597864437)
phasing_angles_4ant = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 11))
phasing_angles_8ant = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 21))


class mySimulation(simulation.simulation):

    def _detector_simulation_filter_amp(self, evt, station, det):
        channelBandPassFilter.run(evt, station, det, passband=[80 * units.MHz, 230 * units.MHz],
                                  filter_type='cheby1', order=4, rp=.1)
        channelBandPassFilter.run(evt, station, det, passband=[0 * units.MHz, 240 * units.MHz],
                                  filter_type='cheby1', order=9, rp=.1)

    def _detector_simulation_part2(self):

        # start detector simulation
        efieldToVoltageConverter.run(self._evt, self._station, self._det)  # convolve efield with antenna pattern

        # downsample trace back to detector sampling rate
        new_sampling_rate = 500.0 * units.MHz
        channelResampler.run(self._evt, self._station, self._det, sampling_rate=new_sampling_rate)

        channelBandPassFilter.run(self._evt, self._station, self._det, passband=[80 * units.MHz, 230 * units.MHz],
                                  filter_type='cheby1', order=4, rp=.1)
        channelBandPassFilter.run(self._evt, self._station, self._det, passband=[0 * units.MHz, 240 * units.MHz],
                                  filter_type='cheby1', order=9, rp=.1)

        window_4ant = int(16 * units.ns * new_sampling_rate * 2.0)
        step_4ant = int(8 * units.ns * new_sampling_rate * 2.0)

        window_8ant = int(16 * units.ns * new_sampling_rate * 4.0)
        step_8ant = int(8 * units.ns * new_sampling_rate * 4.0)

        Vrms = self._Vrms

        simpleThreshold.run(self._evt, self._station, self._det,
                            threshold=3.5 * Vrms,
                            triggered_channels=[4],  # run trigger on all channels
                            number_concidences=1,
                            trigger_name='dipole_1.0sigma')
        simpleThreshold.run(self._evt, self._station, self._det,
                            threshold=3.0 * Vrms,
                            triggered_channels=[4],  # run trigger on all channels
                            number_concidences=1,
                            trigger_name='dipole_1.5sigma')
        simpleThreshold.run(self._evt, self._station, self._det,
                            threshold=2.5 * Vrms,
                            triggered_channels=[4],  # run trigger on all channels
                            number_concidences=1,
                            trigger_name='dipole_2.0sigma')
        simpleThreshold.run(self._evt, self._station, self._det,
                            threshold=2.0 * Vrms,
                            triggered_channels=[4],  # run trigger on all channels
                            number_concidences=1,
                            trigger_name='dipole_2.5sigma')
        simpleThreshold.run(self._evt, self._station, self._det,
                            threshold=1.5 * Vrms,
                            triggered_channels=[4],  # run trigger on all channels
                            number_concidences=1,
                            trigger_name='dipole_3.0sigma')
        simpleThreshold.run(self._evt, self._station, self._det,
                            threshold=1.0 * Vrms,
                            triggered_channels=[4],  # run trigger on all channels
                            number_concidences=1,
                            trigger_name='dipole_3.5sigma')

        # After all of those have fired away, now we need to introduce noise to the signal since noise will change the overall power in an integration window
        original_traces = {}
        for channel in self._station.iter_channels():  # loop over all channels (i.e. antennas) of the station
            original_traces[channel.get_id()] = np.array(channel.get_trace())

        min_freq = 0.0 * units.MHz
        max_freq = 250.0 * units.MHz
        fff = np.linspace(min_freq, max_freq, 10000)
        filt1_highres = channelBandPassFilter.get_filter(fff, 0, 0, None, passband=[0, 240 * units.MHz], filter_type="cheby1", order=9, rp=.1)
        filt2_highres = channelBandPassFilter.get_filter(fff, 0, 0, None, passband=[80 * units.MHz, 230 * units.MHz], filter_type="cheby1", order=4, rp=.1)
        filt_highres = filt1_highres * filt2_highres
        bandwidth = np.trapz(np.abs(filt_highres) ** 2, fff)
        Vrms_ratio = np.sqrt(bandwidth / (max_freq - min_freq))

        # search for noise traces that don't set off a trigger
        has_triggered = True
        while has_triggered:

            for channel in self._station.iter_channels():
                trace = np.zeros(len(original_traces[channel.get_id()][:]))
                channel.set_trace(trace, sampling_rate=new_sampling_rate)

            channelGenericNoiseAdder.run(self._evt, self._station, self._det, amplitude=Vrms / Vrms_ratio,
                                         min_freq=min_freq, max_freq=max_freq, type='rayleigh')

            # bandpass filter trace, the upper bound is higher then the sampling rate which makes it just a highpass filter
            channelBandPassFilter.run(self._evt, self._station, self._det, passband=[0 * units.MHz, 240 * units.MHz],
                                      filter_type='cheby1', order=9, rp=.1)
            channelBandPassFilter.run(self._evt, self._station, self._det, passband=[80 * units.MHz, 230 * units.MHz],
                                      filter_type='cheby1', order=4, rp=.1)

            has_triggered = triggerSimulator.run(self._evt, self._station, self._det,
                                                 Vrms=Vrms,
                                                 threshold=1.77 * np.power(Vrms, 2.0) * window_4ant,
                                                 triggered_channels=range(4, 8),
                                                 phasing_angles=phasing_angles_4ant,
                                                 ref_index=1.75,
                                                 trigger_name='4ant_testing_phasing',  # the name of the trigger
                                                 trigger_adc=False,  # Don't have a seperate ADC for the trigger
                                                 adc_output='voltage',  # output in volts
                                                 trigger_filter=None,
                                                 upsampling_factor=2,
                                                 window=window_4ant,
                                                 step=step_4ant)

            if(has_triggered):
                print('Trigger on noise... for the 4 antenna setup')

        for channel in self._station.iter_channels():
            noise = channel.get_trace()
            trace = original_traces[channel.get_id()]
            channel.set_trace(trace + noise, sampling_rate=new_sampling_rate)

        # run the 4 phased trigger
        triggerSimulator.run(self._evt, self._station, self._det,
                             Vrms=Vrms,
                             threshold=1.77 * np.power(Vrms, 2.0) * window_4ant,
                             triggered_channels=range(4, 8),  # run trigger on all channels
                             phasing_angles=phasing_angles_4ant,
                             ref_index=1.75,
                             trigger_name='4ant_phasing',  # the name of the trigger
                             trigger_adc=False,  # Don't have a seperate ADC for the trigger
                             adc_output='voltage',  # output in volts
                             trigger_filter=None,
                             upsampling_factor=2,
                             window=window_4ant,
                             step=step_4ant)

        has_triggered = True

        # search for noise traces that don't set off a trigger
        while has_triggered:

            for channel in self._station.iter_channels():
                trace = np.zeros(len(original_traces[channel.get_id()][:]))
                channel.set_trace(trace, sampling_rate=new_sampling_rate)

            # Adding noise AFTER the SNR calculation
            # no adding noise, see what that does to the SNR
            channelGenericNoiseAdder.run(self._evt, self._station, self._det, amplitude=Vrms / Vrms_ratio,
                                         min_freq=min_freq, max_freq=max_freq, type='rayleigh')

            # bandpass filter trace, the upper bound is higher then the sampling rate which makes it just a highpass filter
            channelBandPassFilter.run(self._evt, self._station, self._det, passband=[0 * units.MHz, 240 * units.MHz],
                                      filter_type='cheby1', order=9, rp=.1)
            channelBandPassFilter.run(self._evt, self._station, self._det, passband=[80 * units.MHz, 230 * units.MHz],
                                      filter_type='cheby1', order=4, rp=.1)

            has_triggered = triggerSimulator.run(self._evt, self._station, self._det,
                                                 Vrms=Vrms,
                                                 threshold=1.83 * np.power(Vrms, 2.0) * window_8ant,  # see phased trigger module for explanation
                                                 triggered_channels=None,  # run trigger on all channels
                                                 phasing_angles=phasing_angles_8ant,
                                                 ref_index=1.75,
                                                 trigger_name='8ant_testing_phasing',  # the name of the trigger
                                                 trigger_adc=False,  # Don't have a seperate ADC for the trigger
                                                 adc_output='voltage',  # output in volts
                                                 trigger_filter=None,
                                                 upsampling_factor=4,
                                                 window=window_8ant,
                                                 step=step_8ant)

            if(has_triggered):
                print('Trigger on noise... for the 8 antenna setup')

        # loop over all channels (i.e. antennas) of the station
        for channel in self._station.iter_channels():
            noise = channel.get_trace()
            trace = original_traces[channel.get_id()]
            channel.set_trace(trace + noise, sampling_rate=new_sampling_rate)

        # run the 8 channel phased trigger
        triggerSimulator.run(self._evt, self._station, self._det,
                             Vrms=Vrms,
                             threshold=1.83 * np.power(Vrms, 2.0) * window_8ant,  # see phased trigger module for explanation
                             triggered_channels=None,  # run trigger on all channels
                             phasing_angles=phasing_angles_8ant,
                             ref_index=1.75,
                             trigger_name='8ant_phasing',  # the name of the trigger
                             trigger_adc=False,  # Don't have a seperate ADC for the trigger
                             adc_output='voltage',  # output in volts
                             trigger_filter=None,
                             upsampling_factor=4,
                             window=window_8ant,
                             step=step_8ant)


parser = argparse.ArgumentParser(description='Run NuRadioMC simulation')
parser.add_argument('--inputfilename', type=str,
                    help='path to NuRadioMC input event list', default='0.00_12_00_1.00e+16_1.00e+19.hdf5')
parser.add_argument('--detectordescription', type=str,
                    help='path to file containing the detector description', default='4antennas_100m_1.5GHz.json')
parser.add_argument('--config', type=str,
                    help='NuRadioMC yaml config file', default='config_RNO.yaml')
parser.add_argument('--outputfilename', type=str,
                    help='hdf5 output filename', default='output_PA_RNO.hdf5')
parser.add_argument('--outputfilenameNuRadioReco', type=str, nargs='?', default=None,
                    help='outputfilename of NuRadioReco detector sim file')
args = parser.parse_args()

sim = mySimulation(
    inputfilename=args.inputfilename,
    outputfilename=args.outputfilename,
    detectorfile=args.detectordescription,
    outputfilenameNuRadioReco=args.outputfilenameNuRadioReco,
    config_file=args.config)
sim.run()
