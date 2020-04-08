import argparse
# import detector simulation modules
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.trigger.highLowThreshold
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelGenericNoiseAdder
from NuRadioReco.utilities import units
import numpy as np
from NuRadioMC.simulation import simulation
import matplotlib.pyplot as plt
import os

results_folder = 'results'
if not os.path.exists(results_folder):
    os.mkdir(results_folder)

# initialize detector sim modules
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin(time_resolution=0.2*units.ns)
simpleThreshold = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()
highLowThreshold = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()

channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
calculateAmplitudePerRaySolution = NuRadioReco.modules.custom.deltaT.calculateAmplitudePerRaySolution.calculateAmplitudePerRaySolution()

class mySimulation(simulation.simulation):


    def _detector_simulation(self):

        efieldToVoltageConverter.run(self._evt, self._station, self._det)  # convolve efield with antenna pattern
        # downsample trace to internal simulation sampling rate (the efieldToVoltageConverter upsamples the trace to
        # 20 GHz by default to achive a good time resolution when the two signals from the two signal paths are added)
        channelResampler.run(self._evt, self._station, self._det, sampling_rate=1. / self._dt)


        if bool(self._cfg['noise']):

            max_freq = 0.5 / self._dt
            norm = self._get_noise_normalization(self._station.get_id())  # assuming the same noise level for all stations
            Vrms = self._Vrms / (norm / (max_freq)) ** 0.5  # normalize noise level to the bandwidth its generated for
            channelGenericNoiseAdder.run(self._evt, self._station, self._det, amplitude=Vrms, min_freq=0 * units.MHz,
                                         max_freq=max_freq, type='rayleigh')


        channelBandPassFilter.run(self._evt, self._station, self._det,
                                  passband=[1 * units.MHz, 700 * units.MHz], filter_type="butter", order=10)
        channelBandPassFilter.run(self._evt, self._station, self._det,
                                  passband=[150 * units.MHz, 800 * units.GHz], filter_type="butter", order=8)

        highLowThreshold.run(self._evt, self._station, self._det,
                                    threshold_high=5 * self._Vrms,
                                    threshold_low=-5 * self._Vrms,
                                    coinc_window=40 * units.ns,
                                    triggered_channels=[0, 1, 2, 3],
                                    number_concidences=2,  # 2/4 majority logic
                                    trigger_name='hilo_2of4_5_sigma')
        simpleThreshold.run(self._evt, self._station, self._det,
                            threshold=10 * self._Vrms,
                            triggered_channels=[0, 1, 2, 3],
                            trigger_name='simple_10_sigma')


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

sim = mySimulation(inputfilename=args.inputfilename,
                   outputfilename=args.outputfilename,
                   detectorfile=args.detectordescription,
                   outputfilenameNuRadioReco=args.outputfilenameNuRadioReco,
                   config_file=args.config,
                   default_detector_station=101,
                   default_detector_channel=0)
sim.run()
