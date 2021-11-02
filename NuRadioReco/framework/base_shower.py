from __future__ import absolute_import, division, print_function
import NuRadioReco.framework.parameters as parameters
import NuRadioReco.framework.parameter_serialization
from radiotools import helper as hp, coordinatesystems
import pickle

import logging
logger = logging.getLogger('Shower')


class BaseShower:

    def __init__(self, shower_id=0):
        self._id = shower_id
        self._parameters = {}

    def __setitem__(self, key, value):
        self.set_parameter(key, value)

    def __getitem__(self, key):
        return self.get_parameter(key)

    def get_id(self):
        return self._id

    def get_parameter(self, key):
        if not isinstance(key, parameters.showerParameters):
            logger.error("parameter key needs to be of type NuRadioReco.framework.parameters.showerParameters")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.showerParameters")
        return self._parameters[key]

    def set_parameter(self, key, value):
        if not isinstance(key, parameters.showerParameters):
            logger.error("parameter key needs to be of type NuRadioReco.framework.parameters.showerParameters")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.showerParameters")
        self._parameters[key] = value

    def has_parameter(self, key):
        if not isinstance(key, parameters.showerParameters):
            logger.error("parameter key needs to be of type NuRadioReco.framework.parameters.showerParameters")
            raise ValueError("parameter key needs to be of type NuRadioReco.framework.parameters.showerParameters")
        return key in self._parameters

    def get_axis(self):
        """ 
        Returns shower axis. Axis points towards the shower origin, i.e., is antiparallel to the (primary) particle trajectory. 
        The azimuth and zenith angle has to be set (parameters.showerParameters). 
        """
        if not self.has_parameter(parameters.showerParameters.azimuth) or \
           not self.has_parameter(parameters.showerParameters.zenith):
            logger.error(
                "Azimuth or zenith angle not set! Can not return shower axis.")
            raise ValueError(
                "Azimuth or zenith angle not set! Can not return shower axis.")

        return hp.spherical_to_cartesian(self.get_parameter(parameters.showerParameters.zenith),
                                         self.get_parameter(parameters.showerParameters.azimuth))

    def get_coordinatesystem(self):
        """ 
        Returns radiotools.coordinatesystem.cstrafo. Can be used to transform the radio pulses or the observer coordiates in the shower frame.
        Requieres the shower arrival direction (azimuth and zenith angle) and magnetic field vector (parameters.showerParameters).
        """
        if not self.has_parameter(parameters.showerParameters.azimuth) or \
           not self.has_parameter(parameters.showerParameters.zenith) or \
           not self.has_parameter(parameters.showerParameters.magnetic_field_vector):
            logger.error(
                "Magnetic field vector, azimuth or zenith angle not set! Can not return shower coordinatesystem.")
            raise ValueError(
                "Magnetic field vector, azimuth or zenith angle not set! Can not return shower coordinatesystem.")

        return coordinatesystems.cstrafo(self.get_parameter(parameters.showerParameters.zenith),
                                         self.get_parameter(parameters.showerParameters.azimuth),
                                         self.get_parameter(parameters.showerParameters.magnetic_field_vector))

    def serialize(self):
        data = {'_parameters': NuRadioReco.framework.parameter_serialization.serialize(self._parameters),
                '_id': self._id}
        return pickle.dumps(data, protocol=4)

    def deserialize(self, data_pkl):
        data = pickle.loads(data_pkl)
        if '_id' in data.keys():
            self._id = data['_id']
        else:
            self._id = None
        self._parameters = NuRadioReco.framework.parameter_serialization.deserialize(
            data['_parameters'],
            parameters.showerParameters
        )
