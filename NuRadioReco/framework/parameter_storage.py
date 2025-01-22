import NuRadioReco.framework.parameters

import itertools
import logging
logger = logging.getLogger('NuRadioReco.framework.parameter_storage')


class ParameterStorage:

    def __init__(self, parameter_types):
        self._parameters = {}
        self._parameter_covariances = {}
        if not isinstance(parameter_types, list):
            parameter_types = [parameter_types]
        self._parameter_types = parameter_types

    def add_parameter_type(self, parameter_type):
        self._parameter_types.append(parameter_type)

    def __setitem__(self, key, value):
        self.set_parameter(key, value)

    def __getitem__(self, key):
        return self.get_parameter(key)

    def _check_key(self, key):
        if not isinstance(key, tuple(self._parameter_types)):
            logger.error(f"Parameter {key} needs to be of type {self._parameter_types}")
            raise ValueError(f"Parameter {key} needs to be of type {self._parameter_types}")

    def get_parameter(self, key):
        self._check_key(key)
        return self._parameters[key]

    def has_parameter(self, key):
        self._check_key(key)
        return key in self._parameters

    def set_parameter(self, key, value):
        self._check_key(key)
        self._parameters[key] = value

    def set_parameter_error(self, key, value):
        self._check_key(key)
        self._parameter_covariances[(key, key)] = value ** 2

    def get_parameter_error(self, key):
        self._check_key(key)
        return self._parameter_covariances[(key, key)] ** 0.5

    def remove_parameter(self, key):
        self._check_key(key)
        self._parameters.pop(key, None)

    def get_parameters(self):
        return self._parameters

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
