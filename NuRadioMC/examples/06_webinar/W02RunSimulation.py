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

if __name__ == "__main__":
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
    module. This class is incomplete by design, since it lacks the detector simulation
    functions that controls what the detector does after the electric field arrives
    at the antenna. That allows us to create our own class that inherits from
    the simulation class that we will call mySimulation, and define in it a
    _detector_simulation_filter_amp and _detector_simulation_trigger 
    function with all the characteristics of our detector setup.
    """


    class mySimulation(simulation.simulation):
        """
        
        """

        def _detector_simulation_filter_amp(self, evt, station, det):
            """
            This function defines the signal chain, i.e., typically the filters and amplifiers.
            (The antenna response will be applied automatically using the antenna model defined
            in the detector description.)
            In our case,
            we will only implement a couple of filters, one that acts as a low-pass
            and another one that acts as a high-pass.
            """
            channelBandPassFilter.run(evt, station, det,
                                    passband=[1 * units.MHz, 700 * units.MHz], filter_type="butter", order=10)
            channelBandPassFilter.run(evt, station, det,
                                    passband=[150 * units.MHz, 800 * units.GHz], filter_type="butter", order=8)

        def _detector_simulation_trigger(self, evt, station, det):

            """
            This function defines the trigger
            to know when an event has triggered. NuRadioMC and NuRadioReco support multiple
            triggers per detector. As an example, we will use a high-low threshold trigger
            with a high level of 5 times the noise RMS, and a low level of minus
            5 times the noise RMS, a coincidence window of 40 nanoseconds and request
            a coincidence of 2 out of 4 antennas. We can also choose which subset of
            channels we want to use for triggering (we will use the four channels in
            detector.json) by specifying their channel ids, defined in the detector file.
            It is also important to give a descriptive name to the trigger.
            """
            highLowThreshold.run(evt, station, det,
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
            simpleThreshold.run(evt, station, det,
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
