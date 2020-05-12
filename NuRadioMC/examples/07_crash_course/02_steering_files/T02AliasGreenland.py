"""
This file runs a phased array trigger simulation. The phased array configuration
in this file is similar to one of the proposed ideas for RNO: 4 antennas
at a depth of ~100 m, 30 primary phasing directions. In order to run, we need
a detector file and a configuration file, included in this folder. To run
the code, type:

python T02RunPhasedRNO.py --inputfilename input_neutrino_file.hdf5
--detectordescription detector_file.json --config config_file.yaml
--outputfilename output_NuRadioMC_file.hdf5
--outputfilenameNuRadioReco output_NuRadioReco_file.nur (optional)

The antenna positions can be changed in the detector position. The config file
defines de bandwidth for the noise RMS calculation. The properties of the phased
array can be changed in the current file - phasing angles, triggering channels,
bandpass filter and so on.

The present file uses an ADC in first Nyquist zone. The default detector file
(RNO_phased_100m_0.5GHz.json) uses a 0.5 GHz ADC for trigger, that is upsampled
to 1 GHz (upsampling factor 2). If an analog trigger is preferred, trigger_adc
should be set to False.
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
from NuRadioReco.modules import analogToDigitalConverter
from NuRadioReco.utilities.traceWindows import get_window_around_maximum
from NuRadioReco.utilities import units
from NuRadioMC.simulation import simulation
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

parser = argparse.ArgumentParser(description='Run NuRadioMC simulation')
parser.add_argument('--inputfilename', type=str,
                    default=None,
                    help='path to NuRadioMC input event list')
parser.add_argument('--detectordescription', type=str,
                    default='RNO_dipoles.json',
                    help='path to file containing the detector description')
parser.add_argument('--config', type=str,
                    default='config.yaml',
                    help='NuRadioMC yaml config file')
parser.add_argument('--outputfilename', type=str,
                    help='hdf5 output filename')
parser.add_argument('--outputfilenameNuRadioReco', type=str, nargs='?', default=None,
                    help='outputfilename of NuRadioReco detector sim file')
args = parser.parse_args()

# Defining angles for the phased array
main_low_angle = -50 * units.deg
main_high_angle = 50 * units.deg
phasing_angles = np.arcsin( np.linspace( np.sin(main_low_angle), np.sin(main_high_angle), 30) )

# Diode used for finding the correct time window where the pulse lies
# This is done to avoid spurious noise triggers
diode_passband = (None, 200*units.MHz)
diodeSimulator = NuRadioReco.utilities.diodeSimulator.diodeSimulator(diode_passband)

# The 1st zone thresholds have been calculated for a 0.5 GHz ADC,
# with 8 bits and a noise level of 2 bits.
thresholds = { '1Hz' : 20.0 * units.microvolt,
               '2Hz' : 19.75 * units.microvolt,
               '5Hz' : 19.5 * units.microvolt,
               '10Hz' : 19.25 * units.microvolt }

# The following threshold factor refers to an analog phased array with 3 Gs/s.
# They have to be multiplied by the noise RMS.
threshold_factors = { '1Hz' : 2.20 }

# Nyquist zone is 1
nyquist_zone = 1
# Upsampling factor can be changed to another integer
upsampling_factor = 2
# Noise rate in hertz. 1, 2, 5, and 10 available
noise_rate = 1

# Using ADC for digitising traces
use_adc = True
# Using trigger ADC
trigger_adc = True

class mySimulation(simulation.simulation):

    def _detector_simulation(self):
        # start detector simulation
        efieldToVoltageConverter.run(self._evt, self._station, self._det)  # convolve efield with antenna pattern

        new_sampling_rate = 5 * units.GHz
        channelResampler.run(self._evt, self._station, self._det, sampling_rate=new_sampling_rate)

        # Getting the window where most of the electric field lies
        cut_times = get_window_around_maximum(self._station, diodeSimulator, ratio = 0.01)

        # Bool for checking the noise triggering rate
        check_only_noise = False

        if check_only_noise:

            for channel in self._station.iter_channels():  # loop over all channels (i.e. antennas) of the station
                trace = channel.get_trace() * 0
                channel.set_trace(trace, sampling_rate = new_sampling_rate)

        if self._is_simulate_noise():
            max_freq = 0.5 * new_sampling_rate
            norm = self._get_noise_normalization(self._station.get_id())  # assuming the same noise level for all stations
            Vrms = self._Vrms / (norm / (max_freq)) ** 0.5  # normalize noise level to the bandwidth its generated for
            channelGenericNoiseAdder.run(self._evt, self._station, self._det, amplitude=Vrms, min_freq=0 * units.MHz,
                                         max_freq=max_freq, type='rayleigh')

        # We use the [132; 700] MHz band, which is the one proposed for RNO. We approximate
        # the filter with an 8th-order Butterworth on the lower end and a 10th-order
        # Butterworth on the higher end.

        channelBandPassFilter.run(self._evt, self._station, self._det, passband=[132 * units.MHz, 1150 * units.MHz],
                                  filter_type='butter', order=8)
        channelBandPassFilter.run(self._evt, self._station, self._det, passband=[0, 700 * units.MHz],
                                  filter_type='butter', order=10)

        if trigger_adc:
            threshold = thresholds['{:.0f}Hz'.format(noise_rate)]
        else:
            threshold = threshold_factors['{:.0f}Hz'.format(noise_rate)] * self._Vrms

        """
        The thresholds for the phased array have to be calculated by simulating the
        noise trigger rate. Some scripts can be found in the NuRadioReco examples.
        Remember that these thresholds represent the square root of the average
        power in a trigger window that has to be surpassed in order to trigger
        """
        # Running the phased array trigger with ADC, Nyquist zones and upsampling incorporated
        trig = triggerSimulator.run(self._evt, self._station, self._det,
                             threshold=threshold, # see phased trigger module for explanation
                             triggered_channels=None,  # run trigger on all channels
                             trigger_name='alias_phasing', # the name of the trigger
                             phasing_angles=phasing_angles,
                             ref_index=1.75,
                             cut_times=cut_times,
                             trigger_adc=trigger_adc,
                             upsampling_factor=upsampling_factor,
                             nyquist_zone=nyquist_zone,
                             bandwidth_edge=20*units.MHz)

        if use_adc:
            ADC.run(self._evt, self._station, self._det)

sim = mySimulation( inputfilename=args.inputfilename,
                    outputfilename=args.outputfilename,
                    detectorfile=args.detectordescription,
                    outputfilenameNuRadioReco=args.outputfilenameNuRadioReco,
                    config_file=args.config,
                    default_detector_station=101,
                    default_detector_channel=0 )
sim.run()
