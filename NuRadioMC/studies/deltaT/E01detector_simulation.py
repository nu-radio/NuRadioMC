from __future__ import absolute_import, division, print_function
import argparse
from six import iteritems
# import detector simulation modules
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.ARIANNA.triggerSimulator
import NuRadioReco.modules.triggerSimulator
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.custom.deltaT.calculateAmplitudePerRaySolution
from NuRadioReco.utilities import units
from NuRadioMC.simulation import simulation2 as simulation
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
import logging
import numpy as np
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("runDeltaTStudy")

# initialize detector sim modules
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin(debug=False, time_resolution=1 * units.ns,
                               pre_pulse_time=0 * units.ns, post_pulse_time=0 * units.ns)
calculateAmplitudePerRaySolution = NuRadioReco.modules.custom.deltaT.calculateAmplitudePerRaySolution.calculateAmplitudePerRaySolution()
triggerSimulator = NuRadioReco.modules.triggerSimulator.triggerSimulator()
triggerSimulatorARIANNA = NuRadioReco.modules.ARIANNA.triggerSimulator.triggerSimulator()
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()


class mySimulation(simulation.simulation):


    def _detector_simulation(self):
        calculateAmplitudePerRaySolution.run(self._evt, self._station, self._det)
        # save the amplitudes to output hdf5 file
        # save amplitudes per ray tracing solution to hdf5 data output
        if('max_amp_ray_solution' not in self._mout):
            self._mout['max_amp_ray_solution'] = np.zeros((self._n_events, self._n_antennas, 2)) * np.nan
        ch_counter = np.zeros(self._n_antennas, dtype=np.int)
        for efield in self._station.get_sim_station().get_electric_fields():
            for channel_id, maximum in iteritems(efield[efp.max_amp_antenna]):
                self._mout['max_amp_ray_solution'][self._iE, channel_id, ch_counter[channel_id]] = maximum
                ch_counter[channel_id] += 1 
        
        
        # start detector simulation
        efieldToVoltageConverter.run(self._evt, self._station, self._det)  # convolve efield with antenna pattern
        # downsample trace back to detector sampling rate
        channelResampler.run(self._evt, self._station, self._det, sampling_rate=1. / self._dt)
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
                            station_id=101,
                            outputfilenameNuRadioReco=args.outputfilenameNuRadioReco,
                            config_file=args.config)
sim.run()
