import numpy as np
import logging
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
            channel_id,
            cherenkov_angle
    ):
        delta_Cs = []
        viewing_angles = []
        # loop through all ray tracing solution
        for iS in range(self._raytracer.get_number_of_solutions()):
            for key, value in self._raytracer.get_raytracing_output(iS).items():
                sg[key][iSh, channel_id, iS] = value
            self._launch_vector = self._raytracer.get_launch_vector(iS)
            sg['launch_vectors'][iSh, channel_id, iS] = self._launch_vector
            # calculates angle between shower axis and launch vector
            viewing_angle = radiotools.helper.get_angle(self._shower_axis, self._launch_vector)
            viewing_angles.append(viewing_angle)
            delta_C = (viewing_angle - cherenkov_angle)
            logger.debug('solution {} {}: viewing angle {:.1f} = delta_C = {:.1f}'.format(
                iS, propagation.solution_types[self._raytracer.get_solution_type(iS)], viewing_angle / units.deg, (
                            viewing_angle - cherenkov_angle) / units.deg))
            delta_Cs.append(delta_C)
        return delta_Cs, viewing_angles