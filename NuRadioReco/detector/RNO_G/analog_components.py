import numpy as np
from scipy.interpolate import interp1d
import os
from NuRadioReco.utilities import units
import logging
logger = logging.getLogger('analog_components')


def load_amp_response(amp_type='rno_surface', path=os.path.dirname(os.path.realpath(__file__))):  # use this function to read in log data
    """
    Read out amplifier gain and phase. Currently only examples have been implemented.
    Needs a better structure in the future, possibly with database.
    The hardware response incorporator currently reads in the load amp response.
    If you want to read in the RI function fur your reconstruction it needs to be changed
    in modules/RNO_G/hardweareResponseIncorporator.py l. 52, amp response.
    """
    amp_response = {}
    if amp_type == 'rno_surface':
        ph = os.path.join(path, 'HardwareResponses/surface_chan0_LinA.csv')
        ff = np.loadtxt(ph, delimiter=',', skiprows=7, usecols=0)
        amp_gain_discrete = np.loadtxt(ph, delimiter=',', skiprows=7, usecols=5)
        amp_phase_discrete = np.loadtxt(ph, delimiter=',', skiprows=7, usecols=6)
    elif amp_type == 'iglu':
        ph = os.path.join(path, 'HardwareResponses/iglu_drab_chan0_LinA.csv')
        ff = np.loadtxt(ph, delimiter=',', skiprows=7, usecols=0)
        amp_gain_discrete = np.loadtxt(ph, delimiter=',', skiprows=7, usecols=5)
        amp_phase_discrete = np.loadtxt(ph, delimiter=',', skiprows=7, usecols=6)
    else:
        logger.error("Amp type not recognized")
        return amp_response

    # Convert to GHz and add 20dB for attenuation in measurement circuit
    ff *= units.Hz

    amp_gain_f = interp1d(ff, amp_gain_discrete, bounds_error=False, fill_value=1)
    # all requests outside of measurement range are set to 0

    def get_amp_gain(freqs):
        amp_gain = amp_gain_f(freqs)
        return amp_gain

    # Convert to MHz and broaden range
    amp_phase_f = interp1d(ff, np.unwrap(amp_phase_discrete * units.degree),
                           bounds_error=False, fill_value=0)  # all requests outside of measurement range are set to 0

    def get_amp_phase(freqs):
        amp_phase = amp_phase_f(freqs)
        return np.exp(1j * amp_phase)

    amp_response['gain'] = get_amp_gain
    amp_response['phase'] = get_amp_phase

    return amp_response


def get_available_amplifiers():
    return ['iglu', 'rno_surface']
