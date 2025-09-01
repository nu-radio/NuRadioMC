from NuRadioReco.utilities import signal_processing
import warnings

def get_filter_response(*args, **kwargs):
    warnings.warn('This function is deprecated. Use `NuRadioReco.utilities.signal_processing.get_filter_response` instead.', DeprecationWarning)
    return signal_processing.get_filter_response(*args, **kwargs)
