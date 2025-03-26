from NuRadioReco.utilities import trace_utilities
import warnings


class variableWindowSizeCorrelation:
    """
    Module that calculates the correlation between a data trace and a template trace with variable window size
    """

    def __init__(self):
        warnings.warn(
            f"{self.__class__.__name__} is moved to NuRadioReco.utilities.trace_utilities.get_variable_window_size_correlation. "
            "This class will be removed in a future version.", DeprecationWarning, stacklevel=2)

    def begin(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        return trace_utilities.get_variable_window_size_correlation(*args, **kwargs)