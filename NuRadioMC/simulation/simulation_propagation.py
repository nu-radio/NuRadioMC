import numpy as np
import logging
import time
import NuRadioMC.simulation.simulation_base
import radiotools.helper
from NuRadioMC.SignalProp import propagation
from NuRadioReco.utilities import units

logger = logging.getLogger('NuRadioMC')


class simulation_propagation(NuRadioMC.simulation.simulation_base.simulation_base):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super(simulation_propagation, self).__init__(*args, **kwargs)

    def _calculate_viewing_angles(
            self,
            sg,
            iSh,
            i_channel,
            cherenkov_angle
    ):
        delta_Cs = []
        viewing_angles = []
        # loop through all ray tracing solution
        for iS in range(self._raytracer.get_number_of_solutions()):
            # for key, value in self._raytracer.get_raytracing_output(iS).items():
            #     sg[key][iSh, i_channel, iS] = value
            self._launch_vector = self._raytracer.get_launch_vector(iS)
            # calculates angle between shower axis and launch vector
            viewing_angle = radiotools.helper.get_angle(self._shower_axis, self._launch_vector)
            viewing_angles.append(viewing_angle)
            delta_C = (viewing_angle - cherenkov_angle)
            logger.debug('solution {} {}: viewing angle {:.1f} = delta_C = {:.1f}'.format(
                iS, propagation.solution_types[self._raytracer.get_solution_type(iS)], viewing_angle / units.deg, (
                            viewing_angle - cherenkov_angle) / units.deg))
            delta_Cs.append(delta_C)
        return delta_Cs, viewing_angles

    def _perform_raytracing_for_channel(
            self,
            channel_id,
            pre_simulated,
            ray_tracing_performed,
            shower_energy_sum
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
        x2 = self._det.get_relative_position(
            self._station_id, channel_id
        ) + self._det.get_absolute_position(
            self._station_id
        )
        logger.debug(f"simulating channel {channel_id} at {x2}")

        if self._cfg['speedup']['distance_cut']:
            t_tmp = time.time()
            if not self._distance_cut_channel(
                    shower_energy_sum,
                    self._shower_vertex,
                    x2
            ):
                return False
            self._distance_cut_time += time.time() - t_tmp

        self._raytracer.set_start_and_end_point(self._shower_vertex, x2)
        self._raytracer.use_optional_function('set_shower_axis', self._shower_axis)
        if pre_simulated and ray_tracing_performed and not self._cfg['speedup'][
            'redo_raytracing']:  # check if raytracing was already performed
            if self._cfg['propagation']['module'] == 'radiopropa':
                logger.error('Presimulation can not be used with the radiopropa ray tracer module')
                raise Exception('Presimulation can not be used with the radiopropa ray tracer module')
            sg_pre = self._fin_stations["station_{:d}".format(self._station_id)]
            ray_tracing_solution = {}
            for output_parameter in self._raytracer.get_output_parameters():
                ray_tracing_solution[output_parameter['name']] = sg_pre[output_parameter['name']][
                    self._shower_index, channel_id]
            self._raytracer.set_solution(ray_tracing_solution)
        else:
            self._raytracer.find_solutions()

        if not self._raytracer.has_solution():
            logger.debug("event {} and station {}, channel {} does not have any ray tracing solution ({} to {})".format(
                self._event_group_id, self._station_id, channel_id, self._shower_vertex, x2))
            return False

        return True