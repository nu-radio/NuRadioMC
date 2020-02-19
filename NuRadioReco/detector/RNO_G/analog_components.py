import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
from radiotools import helper as hp
from NuRadioReco.utilities import units, io_utilities
import logging
logger = logging.getLogger('analog_components')


def load_amplifier_response(amp_type='10', path=os.path.dirname(os.path.realpath(__file__))):  # here we use log data
    """
    Read out amplifier gain and phase. Currently only examples have been implemented.
    Needs a better structure in the future, possibly with database.
    """
    amplifier_response = {}
    if amp_type == '10':
        ph = os.path.join(path, 'HardwareResponses/surface_-60dBm_chan0_LogA_20dB.csv')
        amp_gain_discrete = np.loadtxt(ph, delimiter=',', skiprows=7, usecols= (0, 5))
        amp_phase_discrete = np.loadtxt(ph, delimiter=',', skiprows=7, usecols= (0, 6))

    else:
        logger.error("Amp type not recognized")
        return amplifier_response

    # Convert to GHz and add 20dB for attenuation in measurement circuit
    amp_gain_discrete[:, 0] *= units.Hz
    if amp_type == '10':
        amp_gain_discrete[:, 1] += 20


    amp_gain_db_f = interp1d(x= amp_gain_discrete[:, 0], y= amp_gain_discrete[:, 1],
                             bounds_error=False, fill_value=0)  # all requests outside of measurement range are set to 0

    def get_amp_gain(ff):
        amp_gain_db = amp_gain_db_f(ff)
        amp_gain = 10 ** (amp_gain_db / 20.)
        return amp_gain

    # Convert to MHz and broaden range
    amp_phase_discrete[:, 0] *= units.Hz


    amp_phase_f = interp1d(x= amp_phase_discrete[:, 0], y= np.unwrap(np.deg2rad(amp_phase_discrete[:, 1])),
                           bounds_error=False, fill_value=0)  # all requests outside of measurement range are set to 0

    def get_amp_phase(ff):
        amp_phase = amp_phase_f(ff)
        return np.exp(1j * amp_phase)

    amplifier_response['gain'] = get_amp_gain
    amplifier_response['phase'] = get_amp_phase

    return amplifier_response

def load_amp_measurement(amp_measurement='surface_-60dBm_chan0_RI_20dB'):  # here we use Real and Imaginary
    """
    load individual amp measurement from file and buffer interpolation function
    """
    filename = os.path.join(os.path.dirname(__file__), 'HardwareResponses/', amp_measurement+".csv")
    ff = np.loadtxt(filename, delimiter=',', skiprows=7, usecols=(0)) # frequency
    response_re = np.loadtxt(filename, delimiter=',', skiprows=7, usecols=5) # real part
    response_im = np.loadtxt(filename, delimiter=',', skiprows=7, usecols=6) # imaginary part
    response = response_re + response_im * 1j
    phase = np.angle(response)
    gain = abs(response)
    amp_phase_f = interp1d(ff, phase, bounds_error=False, fill_value=0)  # all requests outside of measurement range are set to 0
    amp_gain_f = interp1d(ff, gain, bounds_error=False, fill_value=1)  # all requests outside of measurement range are set to 1
    
    def get_response(ff):
        return amp_gain_f(ff) * np.exp(1j * amp_phase_f(ff))
    
    amp_measurements[amp_measurement] = get_response

# amp responses do not occupy a lot of memory, pre load all responses
amplifier_response = {}
for amp_type in ['10']:
    amplifier_response[amp_type] = load_amplifier_response(amp_type)
amp_measurements = {}  # buffer for amp measurements


def get_amplifier_response(ff, amp_type='10', amp_measurement='surface_-60dBm_chan0_RI_20dB'):
    if(amp_measurement is not None):
        if amp_measurement not in amp_measurements:
            load_amp_measurement(amp_measurement)
        return amp_measurements[amp_measurement](ff)
    elif(amp_type in amplifier_response.keys()):
        return amplifier_response[amp_type]['gain'](ff) * amplifier_response[amp_type]['phase'](ff)
    else:
        logger.error("Amplifier response for type {} not implemented, returning None".format(amp_type))
        return None