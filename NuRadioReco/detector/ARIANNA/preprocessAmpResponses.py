import numpy as np
from scipy.interpolate import interp1d
import os
from radiotools import helper as hp
from NuRadioReco.utilities import units
import logging
import pickle
import scipy.signal as ss
from numpy import pi, diff, unwrap, angle
import matplotlib.pyplot  as plt


logger = logging.getLogger('analog_components')

def save_preprocessed_Amps(response,amp_name):
    output_filename = '{}.pkl'.format(amp_name)
    with open(output_filename, 'wb') as fout:
        logger.info('saving output to {}'.format(output_filename))
        pickle.dump(response, fout, protocol=2)

def checkGroupDelay(h,w,amp_name,path):
    gd = os.path.join(path, 'GROUPDELAY'+amp_name)
    amp_gd_discrete = np.loadtxt(gd, skiprows=3, delimiter=',')[:, 1]*1e9

    group_delay = -diff(unwrap(angle(h))) / diff(w) / (2*np.pi) / units.ns

    plt.plot(w,amp_gd_discrete,label='file')
    plt.plot(w[1:],group_delay,label='calc')
    plt.legend()
    plt.show()

def complexNullResponse():
    path = 'HardwareResponses/300SeriesAmpBoxes/'
    amp_gain_discrete = np.loadtxt(os.path.join(path, 'LOGMAG_NULL.CSV'),
                                   skiprows=3, delimiter=',')
    ph = os.path.join(path, 'PHASE_NULL.CSV')
    amp_phase_discrete = np.loadtxt(ph, skiprows=3, delimiter=',')


    # Convert to GHz and add 60dB/40dB for attenuation in measurement circuit
    amp_gain_discrete[:, 0] *= units.Hz
    ff = amp_gain_discrete[:, 0]

    amp_gain_discrete[:, 1] += 40 # 300 series amps

    amp_gain_db_f = interp1d(amp_gain_discrete[:, 0], amp_gain_discrete[:, 1],
                             bounds_error=False, fill_value=0)  # all requests outside of measurement range are set to 0

    def get_amp_gain(ff):
        amp_gain_db = amp_gain_db_f(ff)
        amp_gain = 10 ** (amp_gain_db / 20.)
        return amp_gain

    amp_phase_discrete[:, 0] *= units.Hz

    amp_phase_f = interp1d(amp_phase_discrete[:, 0], np.unwrap(np.deg2rad(amp_phase_discrete[:, 1])),
                           bounds_error=False, fill_value=0)  # all requests outside of measurement range are set to 0

    def get_amp_phase(ff):
        amp_phase = amp_phase_f(ff)
        return np.exp(1j * amp_phase)

    return get_amp_gain(ff)*get_amp_phase(ff)



def preprocess_300Amp(amp,channel='',path="HardwareResponses/"):
    """
    Read out amplifier gain and phase. Currently only examples have been implemented.
    Needs a better structure in the future, possibly with database.

    Parameters
    ----------
    ff: array
    	array of frequencies to probe the response.
    amp: string
        amp series - box number - and channel. Ex: 300-01
    channel: string
        channel id: ex '1'. Default is None fore general 100 and 200 series amps
    path: string
        directory where amp responses are located. Default is "HardwareResponses/"
    """            
    # HardwareResponses/300SeriesAmpBoxes/Box300-03/

    amplifier_response = {}
    path = 'HardwareResponses/300SeriesAmpBoxes/Box' + amp + '/'
    ch_name = '_CH'+channel+'.CSV'
    amp_gain_discrete = np.loadtxt(os.path.join(path, 'LOGMAG' + ch_name),
                                   skiprows=3, delimiter=',')
    ph = os.path.join(path, 'PHASE'+ch_name)
    amp_phase_discrete = np.loadtxt(ph, skiprows=3, delimiter=',')


    # Convert to GHz and add 60dB/40dB for attenuation in measurement circuit
    amp_gain_discrete[:, 0] *= units.Hz
    ff = amp_gain_discrete[:, 0]

    amp_gain_discrete[:, 1] += 40 # 300 series amps

    amp_gain_db_f = interp1d(amp_gain_discrete[:, 0], amp_gain_discrete[:, 1],
                             bounds_error=False, fill_value=0)  # all requests outside of measurement range are set to 0

    def get_amp_gain(ff):
        amp_gain_db = amp_gain_db_f(ff)
        amp_gain = 10 ** (amp_gain_db / 20.)
        return amp_gain

    # Convert to MHz and broaden range
    amp_phase_discrete[:, 0] *= units.Hz

    amp_phase_f = interp1d(amp_phase_discrete[:, 0], np.unwrap(np.deg2rad(amp_phase_discrete[:, 1])),
                           bounds_error=False, fill_value=0)  # all requests outside of measurement range are set to 0

    def get_amp_phase(ff):
        amp_phase = amp_phase_f(ff)
        return np.exp(1j * amp_phase)

    amp_name = amp + '-0' + channel

    response = {}
    response['freqs'] = ff
    response['response'] = (get_amp_gain(ff)*get_amp_phase(ff))/complexNullResponse()
    amplifier_response[amp_name] = response
    
    #checkGroupDelay(get_amp_gain(ff)*get_amp_phase(ff),ff,ch_name,path)
    save_preprocessed_Amps(amplifier_response,amp_name)


preprocess_300Amp('300-03','0')
preprocess_300Amp('300-03','1')
preprocess_300Amp('300-03','2')
preprocess_300Amp('300-03','3')
preprocess_300Amp('300-03','4')
preprocess_300Amp('300-03','5')
preprocess_300Amp('300-03','6')
preprocess_300Amp('300-03','7')

