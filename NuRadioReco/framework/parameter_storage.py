import logging
logger = logging.getLogger('NuRadioReco.framework.parameter_storage')



class ParameterStorage:

    def __init__(self, parameter_type):
        self._parameters = {}
        self._parameter_covariances = {}
        self._parameter_type = parameter_type

    def __setitem__(self, key, value):
        self.set_parameter(key, value)

    def __getitem__(self, key):
        return self.get_parameter(key)

    def _check_key(self, key):
        if not isinstance(key, self._parameter_type):
            logger.error(f"Parameter {key} needs to be of type {self._parameter_type}")
            raise ValueError(f"Parameter {key} needs to be of type {self._parameter_type}")

    def get_parameter(self, key):
        self._check_key(key)
        return self._parameters[key]

    def has_parameter(self, key):
        self._check_key(key)
        return key in self._parameters.keys()

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
