from __future__ import absolute_import, division, print_function
from NuRadioReco.framework.parameters import showerParameters
import NuRadioReco.framework.parameter_storage
from radiotools import helper as hp, coordinatesystems
import pickle
from NuRadioReco.utilities.io_utilities import _dumps

import logging
logger = logging.getLogger('NuRadioReco.Shower')


class BaseShower(NuRadioReco.framework.parameter_storage.ParameterStorage):

    def __init__(self, shower_id=0):
        super().__init__(showerParameters)
        self._id = shower_id

    def get_id(self):
        return self._id

    def get_axis(self):
        """
        Returns the (shower) axis.

        The axis is antiparallel to the movement of the shower particla and point
        towards the origin of the shower.

        Returns
        -------

        np.array(3,)
            Shower axis

        """
        if not self.has_parameter(showerParameters.azimuth) or \
           not self.has_parameter(showerParameters.zenith):
            logger.error(
                "Azimuth or zenith angle not set! Can not return shower axis.")
            raise ValueError(
                "Azimuth or zenith angle not set! Can not return shower axis.")

        return hp.spherical_to_cartesian(self.get_parameter(showerParameters.zenith),
                                         self.get_parameter(showerParameters.azimuth))

    def get_coordinatesystem(self):
        """
        Returns radiotools.coordinatesystem.cstrafo for shower geometry.

        Can be used to transform the radio pulses or the observer coordiates
        in the shower frame. Requieres the shower arrival direction
        (azimuth and zenith angle) and magnetic field vector (showerParameters).

        Returns
        -------

        radiotools.coordinatesystem.cstrafo

        """
        if not self.has_parameter(showerParameters.azimuth) or \
           not self.has_parameter(showerParameters.zenith) or \
           not self.has_parameter(showerParameters.magnetic_field_vector):
            logger.error(
                "Magnetic field vector, azimuth or zenith angle not set! Can not return shower coordinatesystem.")
            raise ValueError(
                "Magnetic field vector, azimuth or zenith angle not set! Can not return shower coordinatesystem.")

        return coordinatesystems.cstrafo(self.get_parameter(showerParameters.zenith),
                                         self.get_parameter(showerParameters.azimuth),
                                         self.get_parameter(showerParameters.magnetic_field_vector))

    def __str__(self):
        def print_key(key, add=""):
            if self.has_parameter(key):
                return f"{str(self.get_parameter(key))}{add}"
            else:
                return "-"

        return f"Shower {self._id}: E = {print_key(showerParameters.energy, ' eV')}" + \
            f", (zen, azi) = ({print_key(showerParameters.zenith, ' deg')}, {print_key(showerParameters.azimuth, ' deg')})" + \
            f", (x, y, z) = ({print_key(showerParameters.vertex, 'm')})"

    def serialize(self):
        data = NuRadioReco.framework.parameter_storage.ParameterStorage.serialize(self)
        data['_id'] = self._id
        return _dumps(data, protocol=4)

    def deserialize(self, data_pkl):
        data = pickle.loads(data_pkl)
        NuRadioReco.framework.parameter_storage.ParameterStorage.deserialize(self, data)
        self._id = data.get('_id', None)

    def show(self, show_parameters=1, print_stdout=True, **kwargs):
        """
        Print an overview of the structure of the Shower.

        Parameters
        ----------
        show_parameters : int, default: 1
            If > 0, print the parameters stored in the Shower.

        Other Parameters
        ----------------
        print_stdout : bool, optional
            If `True` (default), print the Shower structure to stdout.
            Otherwise, return the string representation

        Returns
        -------
        str_output : str, optional
            A string representation of the Shower structure.

        """
        self_string = [f'Shower({self.get_id()})']

        if show_parameters > 0:
            self_string += ['    Parameters']
            par_string = [f'      {par.name:16s}: {val}'
                for par, val in self.get_parameters().items()]
            self_string += par_string

        output = '\n'.join(self_string)

        if print_stdout:
            print(output)
            return

        return output
