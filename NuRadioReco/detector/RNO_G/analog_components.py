import numpy as np
from scipy.interpolate import interp1d
import os
from NuRadioReco.utilities import units, bandpass_filter
from radiotools import helper as hp
import logging
logger = logging.getLogger('analog_components')


def load_amp_response(amp_type='rno_surface', temp=293.15,
                      path=os.path.dirname(os.path.realpath(__file__))):  # use this function to read in log data
    """
    Read out amplifier gain and phase. 
    
    Currently only examples have been implemented. Needs a better structure in the future, possibly with database.
    The hardware response incorporator currently reads in the load amp response.
    If you want to read in the RI function for your reconstruction it needs to be changed
    in modules/RNO_G/hardweareResponseIncorporator.py l. 52, amp response.
    
    Temperature dependence: the function loads a reference measurement made at room temperature
    and will correct it for the temperature. The correction function is obtained empirically for
    one amplifier of reference (one Surface board and one DRAB + fiber + IGLU chain)
    by studying its gain in a climate chamber at different temperatures.
    
    Parameters
    ----------
    amp_type: string

        * "rno_surface": the surface signal chain
        * "iglu": the in-ice signal chain
        * "phased_array": the additional filter of the phased array channels before going into the phased array trigger. 
        * "dummy_amp": no amplifier (response of 1)

    temp: float (default 293.15K)
        the default temperature in Kelvin that the amplifier response is corrected for
    """

    # definition correction functions: temp in Kelvin, freq in GHz
    # functions defined in temperature range [223.15 K , 323.15 K]

    def surface_correction_func(temp, freqs):
        return 1.0377798029 - 0.00135258197 * (temp - 273.15) + (0.4788208019 - 0.01790064797 * (temp - 273.15)) * (freqs ** 5)

    def iglu_correction_func(temp, freqs):
        return 1.1139014286 - 0.00004392995 * ((temp - 273.15) + 28.8331610295) ** 2 + (0.6301058083 - 0.0208741539 * (temp - 273.15)) * (freqs ** 5)

    amp_response = {}
    correction_function = None
    if amp_type == 'rno_surface':
        ph = os.path.join(path, 'HardwareResponses/surface_chan0_LinA.csv')
        ff = np.loadtxt(ph, delimiter=',', skiprows=7, usecols=0)
        ff *= units.Hz
        amp_gain_discrete = np.loadtxt(ph, delimiter=',', skiprows=7, usecols=5)
        amp_phase_discrete = np.loadtxt(ph, delimiter=',', skiprows=7, usecols=6)
        amp_phase_discrete *= units.degree
        correction_function = surface_correction_func
    elif amp_type == 'iglu':
        ph = os.path.join(path, 'HardwareResponses/iglu_drab_chan0_LinA.csv')
        ff = np.loadtxt(ph, delimiter=',', skiprows=7, usecols=0)
        ff *= units.Hz
        amp_gain_discrete = np.loadtxt(ph, delimiter=',', skiprows=7, usecols=5)
        amp_phase_discrete = np.loadtxt(ph, delimiter=',', skiprows=7, usecols=6)
        amp_phase_discrete *= units.degree
        correction_function = iglu_correction_func
    elif amp_type == 'phased_array':
        ph = os.path.join(path, 'HardwareResponses/ULP-216+_Plus25DegC.s2p')
        ff, S11gain, S21deg, S21gain, S21deg, S12gain, S12deg, S22gain, S22deg = np.loadtxt(ph, comments=['#', '!'], unpack=True)
        ff *= units.MHz
        amp_gain_discrete = hp.dB_to_linear(S21gain)
        amp_phase_discrete = S21deg * units.deg
    elif amp_type == 'dummy_amp':
        def dummy_amp_gain(freqs, temp=None):
            return np.ones_like(freqs)
        
        amp_response['gain'] = dummy_amp_gain
        amp_response['phase'] = np.ones_like
        return amp_response
    elif amp_type =='gen2_sim_2021':
        passband_low = [0, 1000*units.MHz]
        passband_high = [96*units.MHz, 100*units.GHz]
        order_low = 7
        order_high = 4
        def get_amp_gain(freqs, temp=None):
            filt1 = bandpass_filter.get_filter_response(
                frequencies=freqs, passband=passband_low,
                filter_type='cheby1', order=order_low, rp=0.1)
            filt2 = bandpass_filter.get_filter_response(
                frequencies=freqs, passband=passband_high,
                filter_type='cheby1', order=order_high, rp=0.1)
            return np.abs(filt1 * filt2)
        
        def get_amp_phase(freqs):
            filt1 = bandpass_filter.get_filter_response(
                frequencies=freqs, passband=passband_low,
                filter_type='cheby1', order=order_low, rp=0.1)
            filt2 = bandpass_filter.get_filter_response(
                frequencies=freqs, passband=passband_high,
                filter_type='cheby1', order=order_high, rp=0.1)
            return np.angle(filt1 * filt2)
        
        amp_response['gain'] = get_amp_gain
        amp_response['phase'] = get_amp_phase
        return amp_response
    else:
        logger.error("Amp type not recognized")
        return amp_response

    amp_gain_f = interp1d(ff, amp_gain_discrete, bounds_error=False, fill_value=0)
    # all requests outside of measurement range are set to 1

    def get_amp_gain(freqs, temp=temp):
        if(correction_function is not None):
            amp_gain = correction_function(temp, freqs) * amp_gain_f(freqs)
        else:
            amp_gain = amp_gain_f(freqs)
        return amp_gain

    # Convert to MHz and broaden range
    amp_phase_f = interp1d(ff, np.unwrap(amp_phase_discrete),
                           bounds_error=False, fill_value=0)  # all requests outside of measurement range are set to 0

    def get_amp_phase(freqs):
        amp_phase = amp_phase_f(freqs)
        return np.exp(1j * amp_phase)

    amp_response['gain'] = get_amp_gain
    amp_response['phase'] = get_amp_phase

    return amp_response


def get_available_amplifiers():
    return ['iglu', 'rno_surface', 'phased_array', 'dummy_amp']
