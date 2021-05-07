Simulation and configuration
==========

The simulation class and the module of the same name, located in the `simulation <https://github.com/nu-radio/NuRadioMC/tree/master/NuRadioMC/simulation>`__ folder, constitute the heart of NuRadioMC. The simulation module takes the neutrino input files and creates events with them. These events are then processed using the information in the config file and the detector layout specified in the detector JSON file. Finally, the trigger description in a steering file is used to determine whether an event triggers or not.

This page outlines the important aspects of how to operate a NuRadioMC simulation. For a practical example, visit `the webinar example <https://github.com/nu-radio/NuRadioMC/tree/master/NuRadioMC/examples/06_webinar>`__.

Let us begin with a description of the steering files needed to run the simulation then let us discuss a brief outline of the procedure in ``simulation.py``. 

    .. Important:: The description of the simulation module, steering files, and configuration files reflect the status of the master as of July 2020. This needs to be updated after the new looping is approved and merged.

Steering files
----------
A NuRadioMC steering file is the file that describes and runs the simulation. As a small example, we can define a detector with an empty detector description. Every steering file should have a class that inherits from the simulation class, that we can call ``mySimulation``. This class must have a method called ``_detector_simulation``, which uses NuRadioReco modules to simulate the detector response.

To create a simulation instance, we need a NuRadioMC neutrino input file, a detector description JSON file, a YAML file with various configuration settings, the output file name, and optionally the output NuRadioReco file name (nur file).

The following code shows how to run a simulation by creating an instance of mySimulation, who is a child class of simulation, and then run it.

    .. code-block:: Python

        from NuRadioMC.simulation import simulation

        # The paths to the input file, output file, detector file and config file should be defined here
        inputfilename = 'input.hdf5'
        outpufilename = 'output.hdf5'
        detectorfile = 'detector.json'
        config_file = 'config.yaml'

        class mySimulation(simulation.simulation):

            def _detector_simulation(self):
                pass

        sim = mySimulation(inputfilename=inputfilename,
                           outputfilename=outputfilename,
                           detectorfile=detectorfile,
                           config_file=config_file)
                       
        sim.run()

When the simulation child object is initialised, the detector description in the JSON file is loaded via NuRadioReco. Then, when the ``run()`` method is called, the simulation module starts the following process:

    1. It reads the events from the input file, one by one, and assigns weights given by the probability that the neutrino reaches our effective volume
    2. It calculates the ray tracing solutions from the interaction vertices to the channels in each station
    3. Then, for the existing ray tracing solutions, the electric field is calculated using the SignalGen models, taking into account that propagation will modify the SignalGen input parameters 
    4. The detector is simulated with the description provided in ``_detector_simulation``. Usually, at least a conversion to voltage, a filter, and a trigger is applied.

An example of a steering file, complete with detector description, can be found in `examples/06_webinar/W02RunSimulation.py <https://github.com/nu-radio/NuRadioMC/blob/master/NuRadioMC/examples/06_webinar/W02RunSimulation.py>`__.

Config files
----------
The default configuration (or config) file used is the ``config_default.yaml`` file in the simulation folder. If the user wants to use different parameters, it suffices to specify ONLY the parameters they want to be overriden in their own config file. For instance, if the user wants to change only the emission model to ARZ2020, a short file like the following is enough:

    .. code-block:: yaml

        signal:
          model: ARZ2020

The following is a description of the default configuration file and what can be changed therein.

    .. code-block:: yaml

        weights:
            weight_mode: core_mantle_crust # core_mantle_crust: use the three layer earth model, 
            # which considers the different densities of the core, mantle and crust.
            cross_section_type: ctw 
            # neutrino cross section: ghandi : according to Ghandi et al. Phys.Rev.D58:093009,1998,
            # ctw : A. Connolly, R. S. Thorne, and D. Waters, Phys. Rev.D 83, 113009 (2011).
            # csms: A. Cooper-Sarkar, P. Mertsch, S. Sarkar, JHEP 08 (2011) 042

The available options for weight mode are:

    * ``simple``: assuming interaction happens at the surface and approximating the Earth with constant density
    * ``core\_mantle\_crust\_simple``: assuming interaction happens at the surface and approximating the Earth with 3 layers of constant density
    * ``core\_mantle\_crust``: approximating the Earth with 3 layers of constant density, path through Earth to interaction vertex is considered
    * ``PREM``: density of Earth is parameterised as a function of radius, path through Earth to interaction vertex is considered
    * ``None``: all weights are set to 1.

    .. code-block:: yaml

        noise: False  # specify if simulation should be run with or without noise
        # The user must add noise manually in the detector description
        sampling_rate: 5.  # sampling rate in GHz used internally in the simulation. 
        # At the end the waveforms will be downsampled to the sampling rate specified in the 
        # detector description.

        seed: 1235 # This seed is used for the first call to the random library

        # The following parameters are used to filter events that we
        know that
        # they will not trigger, gaining time in the process.
        speedup:
          minimum_weight_cut: 1.e-5 # If the assigned weight is less than this one,
          # the event is skipped
          delta_C_cut: 0.698  # 40 degree. If the difference between viewing angle 
          # and Cherenkov angle (corrected by ray tracing) is larger, the event is skipped
          redo_raytracing: False  # redo ray tracing even if previous calculated ray 
          # tracing solutions are present
          min_efield_amplitude: 2  # the minimum signal amplitude of the efield as a factor 
          # of the noise RMS. If the value is smaller, no detector simulation is 
          # performed. As the vector effective length of antennas is typically less 
          # than 1, this cut does not introduce any bias as long as the value is smaller 
          # than the trigger threshold.
          amp_per_ray_solution: True  # if False, the maximum amplitude for each ray tracing 
          # solution is not calculated
          distance_cut: False # if True, a cut for the vertex-observer distance as a function 
          # of shower energy is applied 
          # (log10(max_dist / m) = intercept + slope * log10(shower_energy / eV))
          # The intercept and the slope below have been obtained from distance 
          # histograms for several shower energy bins. A 10x10 array of 1.5 sigma dipoles 
          # in Greenland was used. The distance cut is a linear fit of the maximum distances 
          # at shower energies around 1~10 PeV with a cover factor of 1.5, or 50%.
          distance_cut_intercept: -12.14 # intercept for the maximum distance cut
          distance_cut_slope: 0.9542 # slope for the maximum distance cut
          # This distance cut is really important when simulating large arrays and
          # secondary interactions

        # The following block specifies the options for ray tracing
        propagation:
          module: analytic
          ice_model: southpole_2015 # There's a range of available models:
          # greenland_simple, mooresbay_simple, southpole_2015, southpole_simple, ARASim_southpole
          attenuation_model: SP1 # SP1 for South Pole, GL1 for Greenland, and
          # MB1 for Moore's Bay
          attenuate_ice: True # if True apply the frequency dependent attenuation due to 
          # propagating through ice. (Note: The 1/R amplitude scaling will be applied in either case.)
          n_freq: 25  # the number of frequencies where the attenuation length is 
          # calculated for. The remaining frequencies will be determined from a linear 
          # interpolation between the reference frequencies. The reference frequencies are 
          # equally spaced over the complet frequency range.
          focusing: False  # if True apply the focusing effect.
          focusing_limit: 2  # the maximum amplification factor of the focusing correction
          n_reflections: 0  # the maximum number of reflections off a reflective layer 
          # at the bottom of the ice layer

        # This block specifies the emission model
        signal:
          model: Alvarez2009 # Alvarez2000, Alvarez2009, ARZ2019, or ARZ2020
          zerosignal: False  # if True, the signal is set to zero. This is useful to 
          # study 'noise' only simulations, which inform about the noise trigger rate
          polarization: auto # can be either 'auto' or 'custom'
          ePhi: 0.  # only used if 'polarization = custom', fraction of ePhi component, 
          # the eTheta component is eTheta = (1 - ePhi**2)**0.5
          shower_type: null # optional argument to only simulate certain shower types. 
          # Arguments can be "had" or "em".

        # The last block specifies noise properties important for the trigger
        trigger:
          noise_temperature: 300  # in Kelvin
          # This noise temperature is then used to calculate the noise RMS
          Vrms: null  # the RMS noise value in volts. Not compatible with 'noise_temperature', 
          # if Vrms is set, 'noise_temperature' must be None

        save_all: False # if True, save all events. Otherwise, NuRadioMC will only
        # save triggering events

Detector description
----------
The detector description used by NuRadioReco must be specified in a JSON detector file. We will explain the most important fields needed in these JSON files.

To write a JSON detector file, first we start with the channels, each one having an antenna. The mandatory parameters for a channel are:

    * ``channel_id``, the ID that will be used internally in NuRadioMC and NuRadioReco
    * ``station_id``, the ID of the station the channel belongs to
    * ``adc_sampling_frequency``, the sampling frequency for the analog-to-digital converter, which can be used for resampling traces.
    * ``ant_type``, the antenna type. For instance, 'bicone_v8' or 'createLPDA_100MHz'.
    * These positions are referred to the altitude, northing, and easting of the associated station.
        * ``ant_position_x``, the x coordinate of the antenna in metres
        * ``ant_position_y``, the y coordinate of the antenna in metres
        * ``ant_position_z``, the z coordinate of the antenna in metres.
    * ``ant_orientation_theta``. For dipoles, this theta marks the zenith of the direction along which the axis is placed. For LPDAs, it marks the boresight, the direction of maximum gain. All angles are in degrees.
    * ``ant_orientation_phi``. The same as ant_orientation_theta, but for azimuth.
    * ``ant_rotation_theta``. The zenith angle of the direction of the tines of the antenna, that must be perpendicular to the direction defined by ant_orientation_theta and ant_orientation_phi.
    * ``ant_rotation_phi``. The same as ant_rotation_theta, but for azimuth.

For instance, a vertical dipole would have ``ant_orientation_theta: 0`` and any ``ant_orientation_phi``. ``ant_rotation_theta`` and ``ant_rotation_phi`` can be any direction perpendicular to the orientation, for instance ``ant_rotation_theta: 90`` and ``ant_rotation_phi: 90``.

Another example: a 45-degree downward-pointing LPDA antenna with the tines on the XZ plane would have ``ant_orientation_theta: 135``, ``ant_orientation_phi: 0``, ``ant_rotation_theta: 45``, ``ant_rotation_phi: 0``.

See the `NuRadioReco documentation <https://nu-radio.github.io/NuRadioReco/pages/detector_database_fields.html>`__ for a more complete explanation of the antenna angles.

The rest of the parameters are either not used yet (they're here for future modules) or they are used by modules not needed for the present example.

After the channels, the stations must be defined. The mandatory parameters for a station are:

* ``pos_altitud``, the altitude of the station in metres
* ``pos_easting``, the easting of the station in metres
* ``pos_northing``, the northing of the station in metres
* ``station_id``, the ID of the station that will be used internally by NuRadioMC and NuRadioReco

One of the advantages of the default_detector_station option given by NuRadioReco is that, if we want to create another station 102 that has a channel setup identical to station 101, we can just define station 102 without any channel associated and then run our simulation using ``default_detector_station=101``. The channels from station 101 will be copied to station 102, and their coordinates will be read as relative to the easting, northing, and altitude of station 102.

We show in the following an example of a JSON detector file.

    .. code-block:: json

        {
            "_default": {},
            "channels": {
                "1": {
                    "adc_id": null,
                    "adc_n_samples": 256,
                    "adc_nbits": null,
                    "adc_sampling_frequency": 2.0,
                    "adc_time_delay": null,
                    "amp_reference_measurement": null,
                    "amp_type": "100",
                    "ant_orientation_phi": 0.0,
                    "ant_orientation_theta": 0.0,
                    "ant_position_x": 0.0,
                    "ant_position_y": 0.0,
                    "ant_position_z": -90,
                    "ant_rotation_phi": 90.0,
                    "ant_rotation_theta": 90.0,
                    "ant_type": "vpol_prototype_50cm_n1.74",
                    "cab_id": "17-09",
                    "cab_length": 5.0,
                    "cab_reference_measurement": null,
                    "cab_time_delay": 19.8,
                    "cab_type": "LMR_400",
                    "channel_id": 0,
                    "commission_time": "{TinyDate}:2017-11-01T00:00:00",
                    "decommission_time": "{TinyDate}:2038-01-01T00:00:00",
                    "station_id": 101
                },
                "2": {
                    "adc_id": null,
                    "adc_n_samples": 256,
                    "adc_nbits": null,
                    "adc_sampling_frequency": 2.0,
                    "adc_time_delay": null,
                    "amp_reference_measurement": null,
                    "amp_type": "100",
                    "ant_orientation_phi": 0.0,
                    "ant_orientation_theta": 0.0,
                    "ant_position_x": 0.0,
                    "ant_position_y": 0.0,
                    "ant_position_z": -92.5,
                    "ant_rotation_phi": 90.0,
                    "ant_rotation_theta": 90.0,
                    "ant_type": "bicone_v8_inf_n1.78",
                    "cab_id": "17-09",
                    "cab_length": 5.0,
                    "cab_reference_measurement": null,
                    "cab_time_delay": 19.8,
                    "cab_type": "LMR_400",
                    "channel_id": 1,
                    "commission_time": "{TinyDate}:2017-11-01T00:00:00",
                    "decommission_time": "{TinyDate}:2038-01-01T00:00:00",
                    "station_id": 101
                },
                "3": {
                    "adc_id": null,
                    "adc_n_samples": 256,
                    "adc_nbits": null,
                    "adc_sampling_frequency": 2.0,
                    "adc_time_delay": null,
                    "amp_reference_measurement": null,
                    "amp_type": "100",
                    "ant_orientation_phi": 0.0,
                    "ant_orientation_theta": 0.0,
                    "ant_position_x": 0.0,
                    "ant_position_y": 0.0,
                    "ant_position_z": -95,
                    "ant_rotation_phi": 90.0,
                    "ant_rotation_theta": 90.0,
                    "ant_type": "bicone_v8_inf_n1.78",
                    "cab_id": "17-09",
                    "cab_length": 5.0,
                    "cab_reference_measurement": null,
                    "cab_time_delay": 19.8,
                    "cab_type": "LMR_400",
                    "channel_id": 2,
                    "commission_time": "{TinyDate}:2017-11-01T00:00:00",
                    "decommission_time": "{TinyDate}:2038-01-01T00:00:00",
                    "station_id": 101
                },
                "4": {
                    "adc_id": null,
                    "adc_n_samples": 256,
                    "adc_nbits": null,
                    "adc_sampling_frequency": 2.0,
                    "adc_time_delay": null,
                    "amp_reference_measurement": null,
                    "amp_type": "100",
                    "ant_orientation_phi": 0.0,
                    "ant_orientation_theta": 0.0,
                    "ant_position_x": 0.0,
                    "ant_position_y": 0.0,
                    "ant_position_z": -97.5,
                    "ant_rotation_phi": 90.0,
                    "ant_rotation_theta": 90.0,
                    "ant_type": "bicone_v8_inf_n1.78",
                    "cab_id": "17-09",
                    "cab_length": 5.0,
                    "cab_reference_measurement": null,
                    "cab_time_delay": 19.8,
                    "cab_type": "LMR_400",
                    "channel_id": 3,
                    "commission_time": "{TinyDate}:2017-11-01T00:00:00",
                    "decommission_time": "{TinyDate}:2038-01-01T00:00:00",
                    "station_id": 101
                }
            },
            "positions": {},
            "stations": {
                "1": {
                    "MAC_address": "0002F7F2E7B9",
                    "MBED_type": "v1",
                    "board_number": 203,
                    "commission_time": "{TinyDate}:2017-11-04T00:00:00",
                    "decommission_time": "{TinyDate}:2038-01-01T00:00:00",
                    "pos_altitude": 0,
                    "pos_easting": 0,
                    "pos_measurement_time": null,
                    "pos_northing": 0,
                    "pos_position": "MB1",
                    "pos_site": "mooresbay",
                    "position": "MB1",
                    "station_id": 101,
                    "station_type": null
                }
            }
        }

Detector simulation
----------
The following steering file teaches what the principal constituents of a detector simulation are. This code has been taken from the `examples/06_webinar/W02RunSimulation.py <https://github.com/nu-radio/NuRadioMC/blob/master/NuRadioMC/examples/06_webinar/W02RunSimulation.py>`__. It contains guidelines for generating noise, resampling, filtering, and implementing a trigger.

    .. code-block:: Python

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
        _detector_simulation function with all the characteristics of our detector setup.
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
                at all in radio experiments after the wave has been digitised, with
                sampling rates around the gigahertz. However, we are going to suppose
                that our trigger is analog, so it sees a continuous waveform.

                If the trigger were digital and we needed a sampling rate of, for instance,
                2 GHz, which is what is specified in the detector file, we could use the
                channelResampler module to perform a downsampling as follows:

                new_sampling_rate = self._sampling_rate_detector
                channelResampler.run(self._evt, self._station, self._det, sampling_rate=new_sampling_rate)

                In this case, we are just going to use the resampler with the same sampling
                rate as the simulation, which will leave the trace intact, but we will
                do it for illustration purposes.
                """
                new_sampling_rate = 1 / self._dt
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
                will not be calculated to save time. Thus, we only simulate noise and calculate
                the full trigger for events which have a good chance of triggering. This largely
                reduces the chance of randomly triggering on a thermal noise fluctuation.

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
                Once the signal has been completely processed, we need no define a triggerf
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
        channels with incomplete information and set default_detector_channel=0 - the
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