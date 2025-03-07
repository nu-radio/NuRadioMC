import numpy as np
import os
import logging
from NuRadioReco.utilities import units
from scipy.interpolate import interp1d
from radiotools import helper as hp

logger = logging.getLogger("NuRadioReco.LOFAR.analog_components")


def load_cable_response(cable_length):
    """
    Parameters
    ----------
    cable_length: int or float
        length of the coax cable of the corresponding channel

    Returns
    -------
    default: dict
        A dictionary containing the cable attenuation values for the specified cable length.
        The dictionary has the following keys:

        - 'frequencies': An array of frequency values ranging from 30 MHz to 80 MHz.
        - 'attenuation': An array of cable attenuation values corresponding to the frequencies.
    """
    module_dir = os.path.dirname(__file__)
    file_path = os.path.join(module_dir, "signalchain", f"attenuation_RG58_{int(cable_length)}m.txt")

    data = np.loadtxt(file_path)

    default = {
        'frequencies': np.arange(30, 81) * units.MHz,
        'attenuation': -1 * data
    }
    return default

def get_cable_response(frequencies, cable_length):
    """
    Calculate the cable response for given frequencies and cable length.

    This function loads the LOFAR cable response based on the provided cable length,
    interpolates the attenuation values for the specified frequencies, and returns
    the interpolated cable response.

    Parameters
    ----------
    frequencies: array-like
        An array of frequency values for which the cable response is to be calculated.
    cable_length: int or float
        The length of the cable for which the response is to be loaded.

    Returns
    -------
    cable: dict
        A dictionary containing the interpolated cable attenuation values for the specified frequencies.
        The dictionary has the following key:

        - 'attenuation': An array of interpolated attenuation values corresponding to the input frequencies.
    """

    cable_response = load_cable_response(cable_length=cable_length)
    orig_frequencies = cable_response['frequencies']
    gain = cable_response['attenuation']

    interp_gain = interp1d(orig_frequencies, gain[:,1], bounds_error=False, fill_value=0)

    cable = {
        'attenuation': interp_gain(frequencies)
    }
    return cable


def load_RCU_response():
    """
    Load the RCU (Receiver Control Unit) response data.

    This function reads the RCU gain data from a text file located in the "signalchain" directory
    relative to the current module's directory. It then constructs a dictionary containing the
    frequency range and the corresponding gain values.

    Returns
    -------
    rcu_response: dict
        A dictionary containing the RCU response data with the following keys:

        - 'frequencies': An array of frequency values ranging from 30 MHz to 80 MHz.
        - 'gain': An array of gain values in dB corresponding to the frequencies.
    """

    module_dir = os.path.dirname(__file__)
    file_path = os.path.join(module_dir, "signalchain/RCU_gain.txt")

    data = np.loadtxt(file_path)

    rcu_response = {
        'frequencies': np.arange(30, 81) * units.MHz,
        'gain': data
    }

    return rcu_response

def get_RCU_response(frequencies):
    """
    Fetches the RCU response for given frequencies.

    This function loads the LOFAR RCU response, interpolates the gain values for the specified
    frequencies, and returns the interpolated RCU response.

    Parameters
    ----------
    frequencies: array-like
        An array of frequency values for which the RCU response is to be calculated.

    Returns
    -------
    system: dict
        A dictionary containing the interpolated RCU gain values for the specified frequencies.
        The dictionary has the following key:

        - 'gain': An array of interpolated gain values corresponding to the input frequencies.
    """

    RCU_response = load_RCU_response()
    orig_frequencies = RCU_response['frequencies']
    gain = RCU_response['gain']

    interp_gain = interp1d(orig_frequencies, gain, bounds_error=False, fill_value=0)

    system = {'gain': interp_gain(frequencies)}

    return system
