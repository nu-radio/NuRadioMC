from NuRadioReco.utilities import trace_utilities
import logging
import warnings


class variableWindowSizeCorrelation:
    """
        Module that calculates the correlation between a data trace and a template trace with variable window size
    """

    def __init__(self):
        warnings.warn(f"{self.__class__.__name__} is moved to NuRadioReco.utilities.trace_utilities.get_variable_window_size_correlation. This class will be removed in a future version.", DeprecationWarning, stacklevel=2)
        self.__debug = None
        self.logger = logging.getLogger('NuRadioReco.utilities.variableWindowSizeCorrelation')
        self.begin()

    def begin(self, debug=False, logger_level=logging.NOTSET):
        """
        begin method

        initialize variableWindowSizeCorrelation

        Parameters
        ----------
        debug: boolean
            if true, debug information and plots will be printed
        logger_level: string or logging variable
            Set verbosity level for logger (default: logging.NOTSET)
        """

        self.__debug = debug
        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logger_level)

    def run(self, *args, **kwargs):
        return trace_utilities.get_variable_window_size_correlation(*args, **kwargs)