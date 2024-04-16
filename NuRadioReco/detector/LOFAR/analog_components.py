import numpy as np
import os
import logging
from NuRadioReco.utilities import units
from scipy.interpolate import interp1d
from radiotools import helper as hp

logger = logging.getLogger("NuRadioReco.LOFAR_analog_components")


def load_cable_response(cable_length, path="signalchain"):
    """
    Parameters
    ----------
    path: string
        Path to the folder containing the cable response
    cable_length: int
        length of the coax cable of the corresponding channel
    """

    data = np.loadtxt(f"{path}/attenuation_RG58_{cable_length}m.txt")
    default = {}
    default['frequencies'] = np.arange(30,81) * units.MHz
    default['gain'] = hp.dB_to_linear(data)   #data is in dB
    return default

def get_cable_response(frequencies, cable_length):

    cable_response = {}
    cable_response['default'] = load_cable_response(cable_length=cable_length)

    orig_frequencies = cable_response['default']['frequencies']
    gain = cable_response['default']['gain']

    interp_gain = interp1d(orig_frequencies, gain[:,1], bounds_error=False, fill_value=0)

    cable = {}
    cable['gain'] = interp_gain(frequencies)
    return cable


def load_RCU_response(path="signalchain"):
    """
    Parameters
    ----------
    path: string
        Path to the folder containing the RCU response
    """

    data = np.loadtxt(f"{path}/RCU_gain_new_5.txt")
    default = {}
    default['frequencies'] = np.arange(30,81) * units.MHz
    default['gain'] = hp.dB_to_linear(data)   # data is in dB

    return default

def get_RCU_response(frequencies):


    RCU_response = {}
    RCU_response['default'] = load_RCU_response()
    orig_frequencies = RCU_response['default']['frequencies']

    gain = RCU_response['default']['gain']

    interp_gain = interp1d(orig_frequencies, gain[:,1], bounds_error=False, fill_value=0)

    system = {}
    system['gain'] = interp_gain(frequencies)

    return system
