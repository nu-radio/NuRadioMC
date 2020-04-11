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

"""
This file is a steering file that runs a simple NuRadioMC simulation. If one
wants to run it with the default parameters, one just needs to type:

python W02RunSimulation.py

Otherwise, the arguments need to be specified as follows:

python W02RunSimulation.py --inputfilename input.hdf5 --detectordescription detector.json
--config config.yaml --outputfilename out.hdf5 --outputfilenameNuRadioReco out.nur

The last argument is optional, only needed if the user wants a nur file. nur files
contain lots of information on triggering events, so they're a great tool for
reconstruction (see NuRadioReco documentation and Christoph's webinar). However,
because of their massive amount of information, they can be really heavy. So, when
running NuRadioMC with millions of events, most of the time nur files should not
be created.

Be sure to read the comments in the config.yaml file and also the file
comments_detector.txt to understand how the detector.json function is structured.
"""

parser = argparse.ArgumentParser(description='Run NuRadioMC simulation')
parser.add_argument('--inputfilename', type=str, default='input_3.2e+19_1.0e+20.hdf5',
                    help='path to NuRadioMC input event list')
parser.add_argument('--detectordescription', type=str, default='detector.json',
                    help='path to file containing the detector description')
parser.add_argument('--config', type=str, default='config.yaml',
                    help='NuRadioMC yaml config file')
parser.add_argument('--outputfilename', type=str, default=os.path.join(results_folder, 'NuMC_output.hdf5'),
                    help='hdf5 output filename')
parser.add_argument('--outputfilenameNuRadioReco', type=str, nargs='?', default=None,
                    help='outputfilename of NuRadioReco detector sim file')
args = parser.parse_args()

"""
First we initialise the modules we are going to use. For our simulation, we are
going to need the following ones, which are explained below.
"""
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
simpleThreshold = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()
highLowThreshold = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()

"""
A typical NuRadioMC simulation uses the simulation class from the simulation
module. This class is incomplete by design, since it lacks the _detector_simulation
function that controls what the detector does after the electric field arrives
at the antenna. That allows us to create our own class that inherits from
the simulation class that we will call mySimulation, and define in it a
_detector_simulation class with all the characteristics of our detector setup.
"""

class mySimulation(simulation.simulation):

    def _detector_simulation(self):

        """
        First we convolve the electric field with the antenna pattern to obtain
        the voltage at the antenna terminals. This is done by the efieldtoVoltageConverter.
        """
        efieldToVoltageConverter.run(self._evt, self._station, self._det)
        """
        Our simulation uses the default sampling rate of 5 GHz, or 5 GS/s, or
        equivalently, a time step of 0.2 ns. Such a high resolution, while needed
        during simulations to capture all the details of the radio wave, is not common
        at all in radio experiments, where the sampling rates tend to lie around
        the gigahertz. If we want our simulation to be a theoretical study, we
        can use 5 GHz as the electric field sampling rate. However, if we want
        to simulate what an actual radio detector would see,
        we must resample to lower sampling rates, to the actual sampling rate of
        our analog-to-digital converter. We have specified in our detector.json
        file an adc_sampling_frequency of 2 GHz, which can be accessed using the
        property _sampling_rate_detector.
        """
        new_sampling_rate = self._sampling_rate_detector
        channelResampler.run(self._evt, self._station, self._det, sampling_rate=new_sampling_rate)

        """
        If our config file has specified 'noise: True', this steering file will add
        noise to the simulation. Keep in mind that adding noise can cause some
        events to trigger on noise, while they should not be triggering at all.
        This problem is partially mitigated by a speed-up cut that can be
        controlled with the config file. As default, we have:
        speedup:
            min_efield_amplitude: 2
        This means that if the electric field amplitude is less than twice the
        noise voltage RMS (assuming an antenna effective height of 1), the trigger
        will not be calculated to save time.

        This is a typical problem with detectors. The solution would be to find
        a threshold to trigger on as many signals as possible while keeping the
        noise trigger rate as low as possible. This can be studied setting
        'signal: zerosignal: True' in the yaml config file. The detector will
        try to trigger on noise only and that will give an estimate on the noise
        trigger rate and how many events are not triggering on signal.
        """
        if self._is_simulate_noise():

            """
            The noise level depends on the bandwidth, so we must specify a correct
            level for our bandwidth. Fortunately, NuRadioMC offers a convenient
            solution. We can generate noise for the band [0; new_sampling_rate/2]
            and then use the detector filters to get the actual noise we would
            have at the end of our electronics chain. First, we set the maximum
            frequency.
            """
            max_freq = 0.5 * new_sampling_rate
            """
            Then, we use the function _get_noise_normalization from the simulation
            class, which gives us the effective bandwidth for our detector taking
            into account antenna, filters, and other electronic components.
            """
            det_bandwidth = self._get_noise_normalization(self._station.get_id())
            """
            After that, we calculate the noise level for the [0; max_freq] band,
            which is given by the noise RMS in the actual detector band (self._Vrms,
            calculated by NuRadioMC), and then divided by the square root of
            the actual detector bandwidth and the extended [0; max_freq] bandwidth.
            Remember that the noise RMS formula is
            noise_RMS = sqrt( k_B * T * R * bandwidth ),
            with k_B the Boltzmann constant, T the effective system temperature,
            and R the output resistance.
            """
            Vrms = self._Vrms / (det_bandwidth / max_freq) ** 0.5
            """
            We can now use the channelGenericNoiseAdder, with Rayleigh noise, for
            instance. This module creates noise in a window-like bandwidth, with
            a sharp cut at the edges.
            """
            channelGenericNoiseAdder.run(self._evt, self._station, self._det, amplitude=Vrms,
                                         min_freq=0 * units.MHz,
                                         max_freq=max_freq, type='rayleigh')

        """
        After the signal has been converted to voltage, downsampled, and the noise
        has been added, we can apply the rest of the electronics chain. In our case,
        we will only implement a couple of filters, one that acts as a low-pass
        and another one that acts as a high-pass.
        """
        channelBandPassFilter.run(self._evt, self._station, self._det,
                                  passband=[1 * units.MHz, 700 * units.MHz], filter_type="butter", order=10)
        channelBandPassFilter.run(self._evt, self._station, self._det,
                                  passband=[150 * units.MHz, 800 * units.GHz], filter_type="butter", order=8)

        """
        Once the signal has been completely processed, we need no define a trigger
        to know when an event has triggered. NuRadioMC and NuRadioReco support multiple
        triggers per detector. As an example, we will use a high-low threshold trigger
        with a high level of 5 times the noise RMS, and a low level of minus
        5 times the noise RMS, a coincidence window of 40 nanoseconds and request
        a coincidence of 2 out of 4 antennas. We can also choose which subset of
        channels we want to use for triggering (we will use the four channels in
        detector.json) by specifying their channel ids, defined in the detector file.
        It is also important to give a descriptive name to the trigger.
        """
        highLowThreshold.run(self._evt, self._station, self._det,
                             threshold_high=5 * self._Vrms,
                             threshold_low=-5 * self._Vrms,
                             coinc_window=40 * units.ns,
                             triggered_channels=[0, 1, 2, 3],
                             number_concidences=2,  # 2/4 majority logic
                             trigger_name='hilo_2of4_5_sigma')
        """
        We can add as well a simple trigger threshold of 10 sigma, or 10 times
        the noise RMS. If the absolute value of the voltage goes above that
        threshold, the event triggers.
        """
        simpleThreshold.run(self._evt, self._station, self._det,
                            threshold=10 * self._Vrms,
                            triggered_channels=[0, 1, 2, 3],
                            trigger_name='simple_10_sigma')

"""
Now that the detector response has been written, we create an instance of
mySimulation with the following arguments:
- The input file name, with the neutrino events
- The output file name
- The name of detector description file
- The name of the output nur file (can be None if we don't want nur files)
- The name of the config file

We have also used here two optional arguments, which are default_detector_station
and default_detector channel. If we define a complete detector station with all
of its channels (101 in our case) and we want to add more stations, we can define
these with fewer parameters than needed. Then, making default_detector_station=101,
all the missing necessary parameters will be taken from the station 101, along
with all of the channels from station 101. A similar thing happens if we define
channels with incomplete information and set default_detector_station=0 - the
incomplete channels will be completed using the characteristics from channel 0.
"""
sim = mySimulation(inputfilename=args.inputfilename,
                   outputfilename=args.outputfilename,
                   detectorfile=args.detectordescription,
                   outputfilenameNuRadioReco=args.outputfilenameNuRadioReco,
                   config_file=args.config,
                   default_detector_station=101,
                   default_detector_channel=0)
sim.run()
