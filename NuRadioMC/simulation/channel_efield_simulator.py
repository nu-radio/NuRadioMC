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
            sampling_rate
    ):
        self.__detector = detector
        self.__raytracer = raytracer
        self.__channel_ids = channel_ids
        self.__config = config
        self.__input_data = input_data
        self.__input_attributes = input_attributes
        self.__medium = medium
        self.__trace_length = trace_length
        self.__sampling_rate = sampling_rate
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

    def set_event_group(
            self,
            shower_energy_sum
    ):
        self.__shower_energy_sum = shower_energy_sum

    def set_shower(
            self,
            station_id,
            shower_id,
            shower_index,
            evt_pre_simulated,
            evt_ray_tracing_performed,
    ):
        self.__station_id = station_id
        self.__shower_id = shower_id
        self.__vertex_position = np.array([
            self.__input_data['xx'][shower_index],
            self.__input_data['yy'][shower_index],
            self.__input_data['zz'][shower_index]
        ])
        self.__shower_axis = -1. * radiotools.helper.spherical_to_cartesian(
            self.__input_data['zeniths'][shower_index],
            self.__input_data['azimuths'][shower_index]
        )
        if 'simulation_mode' not in self.__input_attributes or self.__input_attributes['simulation_mode'] == 'neutrino':
            self.__shower_type = self.__input_data['shower_type'][shower_index]
        self.__index_of_refraction = self.__medium.get_index_of_refraction(self.__vertex_position)
        self.__shower_energy = self.__input_data['shower_energies'][shower_index]
        self.__shower_index = shower_index
        self.__evt_pre_simulated = evt_pre_simulated
        self.__evt_ray_tracing_performed = evt_ray_tracing_performed
        self.__cherenkov_angle = np.arccos(1. / self.__medium.get_index_of_refraction(self.__vertex_position))
        self.__shower_parameters = {}
        # print('shower type: ', self.__shower_type)
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
        raytracing_solutions = self.__perform_raytracing_for_channel(channel_id)
        if raytracing_solutions is None:
            return [], [], [], [], [], [], [], []
        n_solutions = len(raytracing_solutions)
        launch_vectors = np.zeros((n_solutions, 3))
        viewing_angles = np.zeros(n_solutions)
        delta_Cs = np.zeros(n_solutions)
        delta_Cs[:] = np.nan

        for i_solution, solution in enumerate(raytracing_solutions):
            launch_vectors[i_solution] = self.__raytracer.get_launch_vector(i_solution)
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
        raytracing_output = []
        for i_solution, solution in enumerate(raytracing_solutions):
            raytracing_output.append(self.__raytracer.get_raytracing_output(i_solution))

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
            if 'simulation_mode' not in self.__input_attributes or self.__input_attributes['simulation_mode'] == 'neutrino':
                efield_spectrum, polarization_angle = self.__simulate_neutrino_emission(
                    launch_vectors[i_solution],
                    receive_vectors[i_solution],
                    viewing_angles[i_solution],
                    path_lenghts[i_solution]
                )
                polarization_directions[i_solution] = polarization_angle


            elif self.__input_attributes['simulation_mode'] == 'emitter':
                efield_spectrum, polarization_angle = self.__simulate_pulser_emission(
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
        channel_id: int
            ID of the channel
        pre_simulated: bool
            Specifies if this shower has already been simulated for the same detector
        ray_tracing_performed: bool
            Specifies if a raytracing solution is already available from the input data, and can be used instead
            of resimulating
        shower_energy_sum: float
            The sum of the energies of all sub-showers in the event

        Returns
            boolean: True is a valid raytracing solution was found, False if no solution has been found or the
            channel is too far away to have a realistic chance of seeing the shower.
        -------

        """
        channel_position = self.__detector.get_relative_position(
            self.__station_id, channel_id
        ) + self.__detector.get_absolute_position(
            self.__station_id
        )

        if not self.__distance_cut_channel(
                self.__shower_energy_sum,
                self.__vertex_position,
                channel_position
        ):
            return

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

    def __distance_cut_channel(
            self,
            shower_energy_sum,
            shower_position,
            channel_position
    ):
        """
        Checks if the channel fulfills the distance cut criterium.
        Returns True if the channel is within the maximum distance
        (and should therefore be simulated) and False otherwise

        Parameters
        ----------
        shower_energy_sum: flaot
            sum of the energies of all sub-showers in this event
        x1: array of floats
            position of the shower
        x2: array of floats
            position of the channel

        Returns
        -------

        """
        if not self.__config['speedup']['distance_cut']:
            return True

        distance_cut = self.__get_distance_cut(shower_energy_sum)
        distance = np.linalg.norm(shower_position - channel_position)
        return distance <= distance_cut

    def __get_distance_cut(self, shower_energy):
        if shower_energy <= 0:
            return 100 * units.m
        return max(100 * units.m, 10 ** self.__distance_cut_polynomial(np.log10(shower_energy)))

    def __get_channel_index(self, channel_id):
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
        return pulser_spectrum, None

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




