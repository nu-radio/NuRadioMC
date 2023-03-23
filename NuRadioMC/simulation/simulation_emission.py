import NuRadioMC.simulation.simulation_base
import NuRadioReco.framework.electric_field
import NuRadioMC.SignalGen.askaryan
import NuRadioMC.SignalGen.emitter
import logging
import numpy as np
import time
import radiotools.coordinatesystems
import radiotools.helper
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.utilities import units, fft
from NuRadioMC.SignalProp import propagation

import scipy.constants

logger = logging.getLogger('NuRadioMC')

class simulation_emission(NuRadioMC.simulation.simulation_base.simulation_base):

    def _simulate_radio_emission(
            self,
            sim_shower,
            channel_id,
            viewing_angles,
            iS,
            n_index,
            distance,
            signal_travel_time,
            receive_vector,
            solution_type
            ):
        t_ask = time.time()
        zenith, azimuth = radiotools.helper.cartesian_to_spherical(*receive_vector)
        polarization_direction_at_antenna = None
        if "simulation_mode" not in self._fin_attrs or self._fin_attrs['simulation_mode'] == "neutrino":
            # first consider in-ice showers
            kwargs = {}
            # if the input file specifies a specific shower realization, use that realization
            if self._cfg['signal']['model'] in ["ARZ2019", "ARZ2020"] and "shower_realization_ARZ" in self._fin:
                kwargs['iN'] = self._fin['shower_realization_ARZ'][self._shower_index]
                logger.debug(f"reusing shower {kwargs['iN']} ARZ shower library")
            elif self._cfg['signal']['model'] == "Alvarez2009" and "shower_realization_Alvarez2009" in self._fin:
                kwargs['k_L'] = self._fin['shower_realization_Alvarez2009'][self._shower_index]
                logger.debug(f"reusing k_L parameter of Alvarez2009 model of k_L = {kwargs['k_L']:.4g}")
            else:
                # check if the shower was already simulated (e.g. for a different channel or ray tracing solution)
                if self._cfg['signal']['model'] in ["ARZ2019", "ARZ2020"]:
                    if sim_shower.has_parameter(shp.charge_excess_profile_id):
                        kwargs = {'iN': sim_shower.get_parameter(shp.charge_excess_profile_id)}
                if self._cfg['signal']['model'] == "Alvarez2009":
                    if sim_shower.has_parameter(shp.k_L):
                        kwargs = {'k_L': sim_shower.get_parameter(shp.k_L)}
                        logger.debug(f"reusing k_L parameter of Alvarez2009 model of k_L = {kwargs['k_L']:.4g}")

            spectrum, additional_output = NuRadioMC.SignalGen.askaryan.get_frequency_spectrum(
                self._fin['shower_energies'][self._shower_index],
                viewing_angles[iS],
                self._n_samples,
                self._dt,
                self._fin['shower_type'][self._shower_index],
                n_index,
                distance,
                self._cfg['signal']['model'],
                seed=self._cfg['seed'],
                full_output=True,
                **kwargs
            )
            # save shower realization to SimShower and hdf5 file
            if self._cfg['signal']['model'] in ["ARZ2019", "ARZ2020"]:
                if 'shower_realization_ARZ' not in self._mout:
                    self._mout['shower_realization_ARZ'] = np.zeros(self._n_showers)
                if not sim_shower.has_parameter(shp.charge_excess_profile_id):
                    sim_shower.set_parameter(shp.charge_excess_profile_id, additional_output['iN'])
                    self._mout['shower_realization_ARZ'][self._shower_index] = additional_output['iN']
                    logger.debug(f"setting shower profile for ARZ shower library to i = {additional_output['iN']}")
            if self._cfg['signal']['model'] == "Alvarez2009":
                if 'shower_realization_Alvarez2009' not in self._mout:
                    self._mout['shower_realization_Alvarez2009'] = np.zeros(self._n_showers)
                if not sim_shower.has_parameter(shp.k_L):
                    sim_shower.set_parameter(shp.k_L, additional_output['k_L'])
                    self._mout['shower_realization_Alvarez2009'][self._shower_index] = additional_output['k_L']
                    logger.debug(f"setting k_L parameter of Alvarez2009 model to k_L = {additional_output['k_L']:.4g}")
            self._askaryan_time += (time.time() - t_ask)

            polarization_direction_onsky = self._calculate_polarization_vector()
            cs_at_antenna = radiotools.coordinatesystems.cstrafo(*radiotools.helper.cartesian_to_spherical(*receive_vector))
            polarization_direction_at_antenna = cs_at_antenna.transform_from_onsky_to_ground(
                polarization_direction_onsky)
            logger.debug(
                'receive zenith {:.0f} azimuth {:.0f} polarization on sky {:.2f} {:.2f} {:.2f}, on ground @ antenna {:.2f} {:.2f} {:.2f}'.format(
                    zenith / units.deg, azimuth / units.deg, polarization_direction_onsky[0],
                    polarization_direction_onsky[1], polarization_direction_onsky[2],
                    *polarization_direction_at_antenna))
            eR, eTheta, ePhi = np.outer(polarization_direction_onsky, spectrum)

        elif self._fin_attrs['simulation_mode'] == "emitter":
            # NuRadioMC also supports the simulation of emitters. In this case, the signal model specifies the electric field polarization
            amplitude = self._fin['emitter_amplitudes'][self._shower_index]
            # following two lines used only for few models( not for all)
            emitter_frequency = self._fin['emitter_frequency'][
                self._shower_index]  # the frequency of cw and tone_burst signal
            half_width = self._fin['emitter_half_width'][
                self._shower_index]  # defines width of square and tone_burst signals
            # get emitting antenna properties
            antenna_model = self._fin['emitter_antenna_type'][self._shower_index]
            antenna_pattern = self._antenna_pattern_provider.load_antenna_pattern(antenna_model)
            ori = [self._fin['emitter_orientation_theta'][self._shower_index],
                   self._fin['emitter_orientation_phi'][self._shower_index],
                   self._fin['emitter_rotation_theta'][self._shower_index],
                   self._fin['emitter_rotation_phi'][self._shower_index]]

            # source voltage given to the emitter
            voltage_spectrum_emitter = NuRadioMC.SignalGen.emitter.get_frequency_spectrum(amplitude, self._n_samples, self._dt,
                                                                      self._fin['emitter_model'][self._shower_index],
                                                                      half_width=half_width,
                                                                      emitter_frequency=emitter_frequency)
            # convolve voltage output with antenna response to obtain emitted electric field
            frequencies = np.fft.rfftfreq(self._n_samples, d=self._dt)
            zenith_emitter, azimuth_emitter =radiotools.helper.cartesian_to_spherical(*self._launch_vector)
            VEL = antenna_pattern.get_antenna_response_vectorized(frequencies, zenith_emitter, azimuth_emitter, *ori)
            c = scipy.constants.c * units.m / units.s
            eTheta = VEL['theta'] * (-1j) * voltage_spectrum_emitter * frequencies * n_index / c
            ePhi = VEL['phi'] * (-1j) * voltage_spectrum_emitter * frequencies * n_index / c
            eR = np.zeros_like(eTheta)
            # rescale amplitudes by 1/R, for emitters this is not part of the "SignalGen" class
            eTheta *= 1 / distance
            ePhi *= 1 / distance
        else:
            logger.error(f"simulation mode {self._fin_attrs['simulation_mode']} unknown.")
            raise AttributeError(f"simulation mode {self._fin_attrs['simulation_mode']} unknown.")

        if self._debug:
            from matplotlib import pyplot as plt
            fig, (ax, ax2) = plt.subplots(1, 2)
            ax.plot(self._ff, np.abs(eTheta) / units.micro / units.V * units.m)
            ax2.plot(self._tt, fft.freq2time(eTheta, 1. / self._dt) / units.micro / units.V * units.m)
            ax2.set_ylabel("amplitude [$\mu$V/m]")
            fig.tight_layout()
            fig.suptitle("$E_C$ = {:.1g}eV $\Delta \Omega$ = {:.1f}deg, R = {:.0f}m".format(
                self._fin['shower_energies'][self._shower_index], viewing_angles[iS], distance))
            fig.subplots_adjust(top=0.9)
            plt.show()

        electric_field = NuRadioReco.framework.electric_field.ElectricField(
            [channel_id],
            position=self._det.get_relative_position(
                self._sim_station.get_id(),
                channel_id
            ),
            shower_id=self._shower_ids[
                self._shower_index],
            ray_tracing_id=iS
        )
        if iS is None:
            a = 1 / 0
        electric_field.set_frequency_spectrum(np.array([eR, eTheta, ePhi]), 1. / self._dt)
        electric_field = self._raytracer.apply_propagation_effects(electric_field, iS)
        # Trace start time is equal to the interaction time relative to the first
        # interaction plus the wave travel time.
        if hasattr(self, '_vertex_time'):
            trace_start_time = self._vertex_time + signal_travel_time
        else:
            trace_start_time = signal_travel_time

        # We shift the trace start time so that the trace time matches the propagation time.
        # The centre of the trace corresponds to the instant when the signal from the shower
        # vertex arrives at the observer. The next line makes sure that the centre time
        # of the trace is equal to vertex_time + T (wave propagation time)
        trace_start_time -= 0.5 * electric_field.get_number_of_samples() / electric_field.get_sampling_rate()

        electric_field.set_trace_start_time(trace_start_time)
        electric_field[efp.azimuth] = azimuth
        electric_field[efp.zenith] = zenith
        electric_field[efp.ray_path_type] = solution_type
        electric_field[efp.nu_vertex_distance] = distance
        electric_field[efp.nu_viewing_angle] = viewing_angles[iS]
        print(np.sqrt(np.max(np.sum(electric_field.get_trace()**2, axis=0))))
        # apply a simple threshold cut to speed up the simulation,
        # application of antenna response will just decrease the
        # signal amplitude
        candidate_ray =  np.max(np.abs(electric_field.get_trace())) > float(self._cfg['speedup']['min_efield_amplitude']) * \
                self._Vrms_efield_per_channel[self._station_id][channel_id]
        return candidate_ray, polarization_direction_at_antenna

    def _calculate_polarization_angles(
            self,
            sim_shower,
            output_data,
            iSh,
            delta_Cs,
            viewing_angles,
            ray_tracing_performed,
            channel_id,
            n_index
    ):
        i_channel = self._get_channel_index(channel_id)
        candidate_station = False
        for iS in range(self._raytracer.get_number_of_solutions()):  # loop through all ray tracing solution
            # skip individual channels where the viewing angle difference is too large
            # discard event if delta_C (angle off cherenkov cone) is too large
            if np.abs(delta_Cs[iS]) > self._cfg['speedup']['delta_C_cut']:
                logger.debug('delta_C too large, ray tracing solution unlikely to be observed, skipping event')
                continue
            if self._check_if_was_pre_simulated() and ray_tracing_performed and not self._cfg['speedup']['redo_raytracing']:
                sg_pre = self._fin_stations["station_{:d}".format(self._station_id)]
                R = sg_pre['travel_distances'][self._shower_index, i_channel, iS]
                T = sg_pre['travel_times'][self._shower_index, i_channel, iS]
            else:
                R = self._raytracer.get_path_length(iS)  # calculate path length
                T = self._raytracer.get_travel_time(iS)  # calculate travel time
                if R is None or T is None:
                    continue
            self._launch_vector = self._raytracer.get_launch_vector(iS)
            receive_vector = self._raytracer.get_receive_vector(iS)
            # get neutrino pulse from Askaryan module

            candidate_ray, polarization_angle = self._simulate_radio_emission(
                sim_shower,
                channel_id,
                viewing_angles,
                iS,
                n_index,
                R,
                T,
                receive_vector,
                propagation.solution_types[self._raytracer.get_solution_type(iS)]
            )
            if candidate_ray:
                candidate_station = True
        return candidate_station