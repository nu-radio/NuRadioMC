import numpy as np
import os
from NuRadioReco.utilities import units
from scipy.interpolate import interp1d

def load_system_response(path=os.path.dirname(os.path.realpath(__file__))):
    """
    Default file was imported from:
    https://github.com/bhokansonfasig/pyrex/tree/master/pyrex/custom/ara/data

    Parameters
    --------
    frequencies: array
        Frequencies for which the amp gain should be delivered
    """

    data = np.loadtxt(os.path.join(path, "HardwareResponses/ARA_Electronics_TotalGain_TwoFilters.txt"),
                                            skiprows=3,delimiter=',')
    default = {}
    default['frequencies'] = data[:,0]*units.MHz
    default['gain'] = data[:,1] #unitless
    default['phase'] = data[:,2] * units.rad # rad

    return default


system_response = {}
system_response['default'] = load_system_response()

def get_system_response(frequencies):

    orig_frequencies = system_response['default']['frequencies']
    phase = system_response['default']['phase']
    gain = system_response['default']['gain']

    interp_phase = interp1d(orig_frequencies, np.unwrap(phase), bounds_error=False, fill_value=0)
    interp_gain = interp1d(orig_frequencies, gain, bounds_error=False, fill_value=0)

    system = {}
    system['gain'] = interp_gain(frequencies)
    system['phase'] =  np.exp(1j * interp_phase(frequencies))
    return system
