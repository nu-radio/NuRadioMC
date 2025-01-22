import logging
logger = logging.getLogger('NuRadioReco.framework.parameter_storage')



def check_key(func):

    def wrapper(*args):
        key = args[1]
        parameter_class = args[0]._parameter_type
        if not isinstance(key, parameter_class):
            logger.error(f"Parameter {key} needs to be of type {parameter_class}")
            raise ValueError(f"Parameter {key} needs to be of type {parameter_class}")
        return func(*args)

    return wrapper


class ParameterStorage:

    def __init__(self, parameter_type):
        self._parameters = {}
        self._parameter_covariances = {}
        self._parameter_type = parameter_type

    def __setitem__(self, key, value):
        self.set_parameter(key, value)

    def __getitem__(self, key):
        return self.get_parameter(key)

    @check_key
    def get_parameter(self, key):
        return self._parameters[key]

    @check_key
    def has_parameter(self, key):
        return key in self._parameters.keys()

    @check_key
    def set_parameter(self, key, value):
        self._parameters[key] = value

    @check_key
    def set_parameter_error(self, key, value):
        self._parameter_covariances[(key, key)] = value ** 2

    @check_key
    def get_parameter_error(self, key):
        return self._parameter_covariances[(key, key)] ** 0.5

    @check_key
    def remove_parameter(self, key):
        self._parameters.pop(key, None)

    def get_parameters(self):
        return self._parameters
