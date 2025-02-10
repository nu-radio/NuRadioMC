"""
Module that implements the basic structure for storing parameters

This class is not intended to be used directly, but is inherited by all
basic NuRadio data objects (see :doc:`Data Structure </NuRadioReco/pages/event_structure>`) that implement
parameter storage.

"""
import NuRadioReco.framework.parameters

import copy
import itertools
import logging
logger = logging.getLogger('NuRadioReco.framework.parameter_storage')


class ParameterStorage:
    """
    This class is the base class to store parameters and their covariances.

    This class is not
    supposed to be used by a user but only be used by other classes to inherit from. All classes
    which have/should have a "parameter storage" shall inherit from this class. A parameter
    storage is a dictionary-like object which stores parameters and their covariances. As
    keys only enums from defined emum classes are allowed (parameter classes). Which
    parameter class is allowed is defined in the constructor of this class or can be
    modified with the function `add_parameter_type`.
    """

    def __init__(self, parameter_types):
        """
        Parameters
        ----------
        parameter_types : parameter class or list of classes
            The parameter classes are defined in `NuRadioReco.framework.parameters`
        """
        self._parameters = {}
        self._parameter_covariances = {}
        if not isinstance(parameter_types, list):
            parameter_types = [parameter_types]
        self._parameter_types = parameter_types

    def add_parameter_type(self, parameter_type):
        """
        Add a parameter class to the list of allowed parameter classes.

        Parameters
        ----------
        parameter_type : parameter class
            The parameter class is defined in `NuRadioReco.framework.parameters`
        """
        self._parameter_types.append(parameter_type)

    def __setitem__(self, key, value):
        """ Set a parameter """
        self.set_parameter(key, value)

    def __getitem__(self, key):
        """ Get a parameter """
        return self.get_parameter(key)

    def _check_key(self, key):
        """ Raises an error if `key` is not a class member of the parameter classes defined for the object s"""
        if not isinstance(key, tuple(self._parameter_types)):
            logger.error(f"Parameter {key} needs to be of type {self._parameter_types}")
            raise ValueError(f"Parameter {key} needs to be of type {self._parameter_types}")

    def get_parameter(self, key):
        """ Get a parameter """
        self._check_key(key)
        return self._parameters[key]

    def has_parameter(self, key):
        """ Returns `True` if the parameter `key` is present, `False` otherwise """
        self._check_key(key)
        return key in self._parameters

    def set_parameter(self, key, value):
        """ Set a parameter """
        self._check_key(key)
        self._parameters[key] = value

    def set_parameter_error(self, key, value):
        """ Set the error of a parameter """
        self._check_key(key)
        self._parameter_covariances[(key, key)] = value ** 2

    def get_parameter_error(self, key):
        """ Get the error of a parameter """
        self._check_key(key)
        return self._parameter_covariances[(key, key)] ** 0.5

    def has_parameter_error(self, key):
        """ Returns `True` if an uncertainty for the parameter `key` is present, `False` otherwise """
        self._check_key(key)
        return (key, key) in self._parameter_covariances

    def remove_parameter(self, key):
        """ Remove a parameter """
        self._check_key(key)
        self._parameters.pop(key, None)

    def get_parameters(self):
        """ Get all parameters """
        return copy.deepcopy(self._parameters)

    def serialize(self):
        parameters = {str(key): self._parameters[key] for key in self._parameters}
        parameter_covariances = {
            (str(key[0]), str(key[1])): self._parameter_covariances[key]
            for key in self._parameter_covariances}

        data = {
            "_parameters": parameters,
            "_parameter_covariances": parameter_covariances,
            "_parameter_types": [parameter_type.__name__ for parameter_type in self._parameter_types]
        }

        return data


    def deserialize(self, data):
        # for backward compatibility
        if 'parameters' in data:
            data['_parameters'] = data['parameters']

        if 'parameter_covariances' in data:
            data['_parameter_covariances'] = data['parameter_covariances']

        parameters = data["_parameters"]
        parameter_covariances = data.get("_parameter_covariances", {})
        if "_parameter_types" in data:
            parameter_types = [NuRadioReco.framework.parameters.__dict__[parameter_type]
                               for parameter_type in data["_parameter_types"]]
        else:
            parameter_types = self._parameter_types

        for parameter_type in parameter_types:
            for key in parameter_type:
                if str(key) in parameters:
                    self._parameters[key] = parameters[str(key)]

            if len(parameter_covariances):
                for key in itertools.product(parameter_type, parameter_type):
                    if (str(key[0]), str(key[1])) in parameter_covariances:
                        self._parameter_covariances[key] = parameter_covariances[(str(key[0]), str(key[1]))]
