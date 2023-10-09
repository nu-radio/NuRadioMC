import numpy as np
import scipy
import radiotools.helper
import radiotools.coordinatesystems
import NuRadioMC.SignalGen.askaryan
import NuRadioMC.SignalGen.emitter
import NuRadioReco.detector.antennapattern
import NuRadioReco.framework.electric_field
import NuRadioMC.SignalGen.parametrizations
from NuRadioMC.SignalGen.parametrizations import _random_generators as shower_random
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.utilities import units

class channelEfieldSimulator:
    def __init__(
            self,
            detector,
            raytracer,
            channel_ids,
            config,
            input_data,
            input_attributes,
            medium,
            trace_length,
            time_logger
    ):
        """
        Initialize class

        Parameters
        ----------
        detector: NuRadioReco.detector.detector.Detector or NuRadioReco.detector.generic_detector.GenericDetector object
            Object conaining the detector description
        raytracer: NuRadioMC.SignalProp.analyticraytracing.ray_tracing object or similar
            Object that can perform the propagation fromt the shower to the detector. Has to follow the template
            specified in NuRadioMC.SignalProp.propagation_base_class. Options are the analytic raytracer, RadioPropa of the 
            direct line raytracer.
        channel_ids: numpy.array of integers
            An array containing the IDs of the channels that are to be simulated.
        config: dict
            A dictionary containing the settings specified in the configuration yaml file
        input_data: dict
            The data from the input HDF5 file containing the showers to be simulated
        input_attributes: dict 
            The attributes from the input HDF5 file
        medium: NuRadioMC.utilities.medium class
            A class containing the index of refraction profile of the ice. Needs to be compatible with the chosen raytracer
        trace_length: integer
            The number of samples of the electric field and voltage waveforms that are simulated
        time_logger:
            NuRadioMC/simulation.time_logger.timeLogger object
            This class is used to keep track of the simulation progress and print updates in regular intervals.
        """
        self.__detector = detector
        self.__raytracer = raytracer
        self.__channel_ids = channel_ids
        self.__config = config
        self.__input_data = input_data
        self.__input_attributes = input_attributes
        self.__medium = medium
        self.__trace_length = trace_length
        self.__sampling_rate = self.__config['sampling_rate'] * units.GHz
        self.__time_logger = time_logger
        self.__shower_id = None
        self.__vertex_position = None
        self.__shower_axis = None
        self.__shower_energy = None
        self.__shower_energy_sum = None
        self.__shower_index = None
        self.__shower_type = None
        self.__evt_pre_simulated = None
        self.__evt_ray_tracing_performed = None
        self.__cherenkov_angle = None
        self.__station_id = None
        self.__shower_parameters = None
        self.__index_of_refraction = None
        self.__antenna_pattern_provider = NuRadioReco.detector.antennapattern.AntennaPatternProvider()
        if self.__config['speedup']['distance_cut']:
            self.__distance_cut_polynomial = np.polynomial.polynomial.Polynomial(
                self.__config['speedup']['distance_cut_coefficients']
            )
        self.__shower_generators = shower_random
        shower_model = self.__config['signal']['model']
        if shower_model not in shower_random:
            self.__shower_generators[shower_model] = np.random.RandomState(self.__config['seed'])
        self.__is_neutrino_simulation = 'simulation_mode' not in self.__input_attributes or self.__input_attributes['simulation_mode'] == 'neutrino'

    def set_shower(
            self,
            station_id,
            shower_id,
            shower_index,
            evt_pre_simulated,
            evt_ray_tracing_performed,
    ):
        """
        Updates information about the shower that is currently simulated

        Parameters:
        -----------
        station_id: integer
            The ID of the station that is being simulated
        shower_id: integer
            The ID of the shower that is being simulated
        shower_index: integer
            The index of the shower that is being simulated, i.e. its position in the HDF5 input file
        event_pre_simulated: bool
            Specifies if simulation results for this shower are already available from the input file
        event_ray_tracing_performed: bool
            Specifies if raytracing results for this shower are already available from the input file
        """

        self.__station_id = station_id
        self.__shower_id = shower_id
        self.__shower_index = shower_index
        self.__evt_pre_simulated = evt_pre_simulated
        self.__evt_ray_tracing_performed = evt_ray_tracing_performed

        self.__vertex_position = np.array([
            self.__input_data['xx'][shower_index],
            self.__input_data['yy'][shower_index],
            self.__input_data['zz'][shower_index]
        ])
        if self.__is_neutrino_simulation:
            self.__shower_type = self.__input_data['shower_type'][shower_index]
            self.__shower_axis = -1. * radiotools.helper.spherical_to_cartesian(
                self.__input_data['zeniths'][shower_index],
                self.__input_data['azimuths'][shower_index]
            )
            self.__shower_energy = self.__input_data['shower_energies'][shower_index]

        self.__index_of_refraction = self.__medium.get_index_of_refraction(self.__vertex_position)
        self.__cherenkov_angle = np.arccos(1. / self.__medium.get_index_of_refraction(self.__vertex_position))
        self.__shower_parameters = {}
        if self.__config['signal']['model'] in ["ARZ2019", "ARZ2020"] and "shower_realization_ARZ" in self.__input_data:
            self.__shower_parameters['iN'] = self.__input_data['shower_realization_ARZ'][self.__shower_index]
        elif self.__config['signal']['model'] == "Alvarez2009":
            if "shower_realization_Alvarez2009" in self.__input_data:
                self.__shower_parameters['k_L'] = self.__input_data['shower_realization_Alvarez2009'][self.__shower_index]
            """
            TODO: This is commented out to make sure the same random numbers are drawn as in the unit tests.
            else:
                self.__shower_parameters['k_L'] = NuRadioMC.SignalGen.parametrizations.get_Alvarez2009_k_L(
                    False,
                    False,
                    self.__input_data['shower_energies'][self.__shower_index],
                    self.__shower_type.upper()
                )
                # print(self.__shower_parameters['k_L'])
            """

    # TODO: Remove function, it's only here to make sure the same random variables are drawn as in the unit tests
    def set_alvarez_k_L(self):
        """
        This function is only supposed to be temporary. Because the new implementation of the simulation
        code gets rid of some unnecessary calls to this function, different random numbers are generated,
        which causes the tests to fail. This function is only used to generate these additional
        random numbers to make sure the same showers are still being simulated.
        """
        self.__shower_parameters['k_L'] = NuRadioMC.SignalGen.parametrizations.get_Alvarez2009_k_L(
            False,
            False,
            self.__input_data['shower_energies'][self.__shower_index],
            self.__shower_type.upper()
        )

    def simulate_efield_at_channel(
            self,
            channel_id
    ):
        """
        Performs the raytracing and radio emission simulation for a single channel.

        Parameters
        ----------
        channel_id: integer
            The ID of the channel that is being simulated
        
        Returns
        -------
        Tupel of 8 lists or arrays with sizes depending on the number n_rays of raytracing solutions
        Entries are as follows:
            0: List of NuRadioReco.framework.electric_field.ElectricField objects of length n_rays
                E-Field objects containing the simulated waveforms and signal properties
                for each raytracing solution
            1: numpy.array of floats with shape (n_rays, 3)
                Directions (in cartesian coordinates) that the radio signals are emitted in
            2: numpy.array of floats with shape (n_rays, 3)
                Directions (in cartesian coordinates) at which the radio signals arrive at the channel
            3: numpy.array of floats with shape (n_rays,)
                Travel times of the radio signals from the emitter to the channel
            4: numpy.array of floats with shape (n_rays,)
                Distance the radio signals travel from the emitter to the channel
            5: numpy.array of floats with shape (n_rays, 3)
                Direction (in cartesian coordinates) of the polrization vectors of the radio signals
            6: numpy.array of floats with shape (n_rays,)
                Maximum amplitudes of the electric fields
            7: dict
                Output that is given by the raytracer class. See the get_raytracing_output method of
                the used raytracer for details.


        """
        self.__time_logger.start_time('ray tracing')
        raytracing_solutions = self.__perform_raytracing_for_channel(channel_id)
        if raytracing_solutions is None:
            return [], [], [], [], [], [], [], []
        self.__time_logger.stop_time('ray tracing')
        n_solutions = len(raytracing_solutions)
        launch_vectors = np.zeros((n_solutions, 3))
        viewing_angles = np.zeros(n_solutions)
        delta_Cs = np.zeros(n_solutions)
        delta_Cs[:] = np.nan

        for i_solution, solution in enumerate(raytracing_solutions):
            launch_vectors[i_solution] = self.__raytracer.get_launch_vector(i_solution)
            if self.__is_neutrino_simulation:
                viewing_angles[i_solution] = radiotools.helper.get_angle(self.__shower_axis, launch_vectors[i_solution])
                delta_Cs[i_solution] = viewing_angles[i_solution] - self.__cherenkov_angle
        path_lenghts = np.zeros(n_solutions)
        travel_times = np.zeros(n_solutions)
        polarization_directions = np.zeros((n_solutions, 3))
        receive_vectors = np.zeros((n_solutions, 3))
        efield_amplitudes = np.zeros(n_solutions)
        path_lenghts[:] = np.nan
        travel_times[:] = np.nan
        receive_vectors[:] = np.nan
        efield_amplitudes[:] = np.nan
        polarization_directions[:] = np.nan
        channel_index = self.__get_channel_index(channel_id)
        channel_relative_position = self.__detector.get_relative_position(self.__station_id, channel_id)
        efield_objects = []
        raytracing_output = {}
        self.__time_logger.start_time('askaryan')
        for i_solution, solution in enumerate(raytracing_solutions):
            rt_output = self.__raytracer.get_raytracing_output(i_solution)
            for key in rt_output:
                if key not in raytracing_output:
                    raytracing_output[key] = np.full(n_solutions, np.nan)
                raytracing_output[key][i_solution] = rt_output[key]

            if np.abs(delta_Cs[i_solution]) >= self.__config['speedup']['delta_C_cut']:
                continue
            if self.__evt_pre_simulated and self.__evt_ray_tracing_performed and not self.__config['speedup']['redo_raytracing']:
                input_data = self.__input_data['station_{:d}'.formta(self.__station_id)]
                dist = input_data['travel_distances'][self.__shower_index, channel_index, i_solution]
                prop_time = input_data['travel_times'][self.__shower_index, channel_index, i_solution]
            else:
                dist = self.__raytracer.get_path_length(i_solution)
                prop_time = self.__raytracer.get_travel_time(i_solution)
            if dist is None or prop_time is None:
                continue
            path_lenghts[i_solution] = dist
            travel_times[i_solution] = prop_time
            receive_vectors[i_solution] = self.__raytracer.get_receive_vector(i_solution)
            if self.__is_neutrino_simulation:
                efield_spectrum, polarization_angle = self.__simulate_neutrino_emission(
                    launch_vectors[i_solution],
                    receive_vectors[i_solution],
                    viewing_angles[i_solution],
                    path_lenghts[i_solution]
                )
                polarization_directions[i_solution] = polarization_angle


            elif self.__input_attributes['simulation_mode'] == 'emitter':
                efield_spectrum = self.__simulate_pulser_emission(
                    launch_vectors[i_solution],
                    path_lenghts[i_solution]
                )
            else:
                raise AttributeError('simulation mode {} inknown.'.format(self.__input_attributes['simulation_mode']))
            efield = NuRadioReco.framework.electric_field.ElectricField(
                [channel_id],
                position=channel_relative_position,
                shower_id=self.__shower_id,
                ray_tracing_id=i_solution
            )
            efield.set_frequency_spectrum(efield_spectrum, self.__sampling_rate)
            efield = self.__raytracer.apply_propagation_effects(efield, i_solution)
            if 'vertex_time' in self.__shower_parameters.keys():
                trace_start_time = self.__shower_parameters['vertex_time'] + travel_times[i_solution]
            else:
                trace_start_time = travel_times[i_solution]
            trace_start_time -= self.__trace_length / self.__sampling_rate / 2.

            efield.set_trace_start_time(trace_start_time)
            zenith_receiver, azimuth_receiver = radiotools.helper.cartesian_to_spherical(*receive_vectors[i_solution])
            efield[efp.azimuth] = azimuth_receiver
            efield[efp.zenith] = zenith_receiver
            efield[efp.ray_path_type] = self.__raytracer.get_solution_type(i_solution)
            efield[efp.nu_vertex_distance] = path_lenghts[i_solution]
            efield[efp.nu_viewing_angle] = viewing_angles[i_solution]
            efield_objects.append(efield)
            efield_amplitudes[i_solution] = np.max(np.abs(efield.get_trace()))
        self.__time_logger.stop_time('askaryan')
        return efield_objects, launch_vectors, receive_vectors, travel_times, path_lenghts, \
            polarization_directions, efield_amplitudes, raytracing_output

    def __perform_raytracing_for_channel(
            self,
            channel_id
    ):
        """
        Performs the raytracing from the shower vertex to a channel.

        Parameters
        ----------
        channel_id: integer
            ID of the channel
        Returns
            list of dicts
            A list with the raytracing solutions. Details depend on the raytracer used.
            If no solution was found, an empty list is returned.
        -------

        """
        channel_position = self.__detector.get_relative_position(
            self.__station_id, channel_id
        ) + self.__detector.get_absolute_position(
            self.__station_id
        )

        self.__raytracer.set_start_and_end_point(self.__vertex_position, channel_position)
        self.__raytracer.use_optional_function('set_shower_axis', self.__shower_axis)
        if self.__evt_pre_simulated and self.__evt_ray_tracing_performed and not self.__config['speedup'][
            'redo_raytracing']:  # check if raytracing was already performed
            if self.__config['propagation']['module'] == 'radiopropa':
                raise Exception('Presimulation can not be used with the radiopropa ray tracer module')
            sg_pre = self.__input_data["station_{:d}".format(self.__station_id)]
            ray_tracing_solution = {}
            for output_parameter in self.__raytracer.get_output_parameters():
                ray_tracing_solution[output_parameter['name']] = sg_pre[output_parameter['name']][
                    self.__shower_index, channel_id]
            self.__raytracer.set_solution(ray_tracing_solution)
        else:
            self.__raytracer.find_solutions()

        return self.__raytracer.get_results()


    def __get_channel_index(self, channel_id):
        """
        Find the index of a channel in the list of simulated channels

        Parameters
        ----------
        channel_id: integer
            The ID of channel
        Returns
        -------
            integer
            The index of the channel among the simulated channel IDs
        """
        index = self.__channel_ids.index(channel_id)
        if index < 0:
            raise ValueError('Channel with ID {} not found in station {} of detector description!'.format(channel_id, self._station_id))
        return index

    def __simulate_neutrino_emission(
            self,
            launch_vector,
            receive_vector,
            viewing_angle,
            propagation_distance
    ):
        """
        Simulates the emission from a neutrino-induced shower

        Parameters
        ----------
        launch_vector: numpy.array of floats with shape (3,)
            The direction (in cartesian coordinates) into which the radio signal is emitted
        receive_vector: numpy.array of floats with shape (3,)
            The direction (in cartesian coordinates) at which the radio signal arrives at the detector
        viewing_angle: float
            The angle between the shower axis and the direction into which the radio signal is emitted
        propagation_distance: float
            The distance over which the radio signal propagates before it arrives at the detector

        Returns
        -------
        tuple of length 2
        Entries are as follows:
            0: numpy.array of floats with shape (3, n_frequency_bins)
            The spectrum of the radio signal in the 3 polarization components (e_r, e_theta, e_phi)
            1: numpy.array of floats with shape (3,)
            The direction (in cartesian coordinates) of the polarization vector of the radio signal at the detector
        """

        if self.__config['signal']['model'] == 'Alvarez2009' and 'k_L' not in self.__shower_parameters.keys():
            self.set_alvarez_k_L()
        spectrum, additional_output = NuRadioMC.SignalGen.askaryan.get_frequency_spectrum(
            self.__shower_energy,
            viewing_angle,
            self.__trace_length,
            1. / self.__sampling_rate,
            self.__shower_type,
            self.__index_of_refraction,
            propagation_distance,
            self.__config['signal']['model'],
            full_output=True,
            **self.__shower_parameters
        )
        polarization_at_shower = self.__get_polarization_vector(launch_vector)
        cs_at_antenna = radiotools.coordinatesystems.cstrafo(*radiotools.helper.cartesian_to_spherical(*receive_vector))
        polarization_at_antenna = cs_at_antenna.transform_from_onsky_to_ground(polarization_at_shower)
        return np.outer(polarization_at_shower, spectrum), polarization_at_antenna

    def __simulate_pulser_emission(
            self,
            launch_vector,
            propagation_distance
    ):
        """
        Simulates the emission from a radio pulser

        Parameters
        ----------
        launch_vector: numpy.array of floats with shape (3,)
            The direction (in cartesian coordinates) into which the radio signal is emitted
        propagation_distance: float
            The distance over which the radio signal propagates before it arrives at the detector

        Returns
        -------
        numpy.array of floats with shape (3, n_frequency_bins)
            The spectrum of the radio signal in the 3 polarization components (e_r, e_theta, e_phi)
        """
        amplitude = self.__input_data['emitter_amplitudes'][self.__shower_index]
        emitter_frequency = self.__input_data['emitter_frequency'][self.__shower_index]
        emitter_half_width = self.__input_data['emitter_half_width'][self.__shower_index]
        antenna_model = self.__input_data['emitter_antenna_type'][self.__shower_index]
        antenna_pattern = self.__antenna_pattern_provider.load_antenna_pattern(antenna_model)
        pulser_orientation = [
            self.__input_data['emitter_orientation_theta'][self.__shower_index],
            self.__input_data['emitter_orientation_phi'][self.__shower_index],
            self.__input_data['emitter_rotation_theta'][self.__shower_index],
            self.__input_data['emitter_rotation_phi'][self.__shower_index]
        ]
        voltage_spectrum_emitter = NuRadioMC.SignalGen.emitter.get_frequency_spectrum(
            amplitude,
            self.__trace_length,
            1. / self.__sampling_rate,
            self.__input_data['emitter_model'][self.__shower_index],
            half_width=emitter_half_width,
            emitter_frequency=emitter_frequency
        )
        frequencies = np.fft.rfftfreq(self.__trace_length, d=1. / self.__sampling_rate)
        zenith_emitter, azimuth_emitter = radiotools.helper.cartesian_to_spherical(*launch_vector)
        VEL = antenna_pattern.get_antenna_response_vectorized(frequencies, zenith_emitter, azimuth_emitter, *pulser_orientation)
        c = scipy.constants.c * units.m / units.s
        pulser_spectrum = np.zeros((3, frequencies.shape[0]), dtype=np.cdouble)
        pulser_spectrum[1] = VEL['theta'] * (-1j) * voltage_spectrum_emitter * frequencies * self.__index_of_refraction / c
        pulser_spectrum[2] = VEL['phi'] * (-1j) * voltage_spectrum_emitter * frequencies * self.__index_of_refraction / c
        # rescale amplitudes by 1/R, for emitters this is not part of the "SignalGen" class
        pulser_spectrum /= propagation_distance
        return pulser_spectrum

    def __get_polarization_vector(
            self,
            launch_vector
    ):
        if self.__config['signal']['polarization'] == 'auto':
            polarization_direction = np.cross(launch_vector, np.cross(self.__shower_axis, launch_vector))
            polarization_direction /= np.linalg.norm(polarization_direction)
            cs = radiotools.coordinatesystems.cstrafo(*radiotools.helper.cartesian_to_spherical(*launch_vector))
            return cs.transform_from_ground_to_onsky(polarization_direction)
        elif self.__config['signal']['polarization'] == 'custom':
            e_phi = float(self.__config['signal']['ePhi'])
            e_theta = np.sqrt(1. - e_phi**2)
            polarization = np.array([0, e_theta, e_phi])
            return polarization / np.linalg.norm(e_phi)
        else:
            raise ValueError("{} for config.signal.polarization is not a valid option".format(self.__config['signal']['polarization']))




