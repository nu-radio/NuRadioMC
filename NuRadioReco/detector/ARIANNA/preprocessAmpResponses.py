import numpy as np
import os
from radiotools import helper as hp
from NuRadioReco.utilities import units
import logging
import pickle
import scipy.signal as ss
from numpy import pi, diff, unwrap, angle
import matplotlib.pyplot  as plt

logger = logging.getLogger('analog_components')


def save_preprocessed_Amps(response, amp_name):
    output_filename = '{}.pkl'.format(amp_name)
    with open(output_filename, 'wb') as fout:
        logger.info('saving output to {}'.format(output_filename))
        pickle.dump(response, fout, protocol=4)

def preprocess_300Amp(amp, channel, path="HardwareResponses/"):
    """
    preprocess individual 300 series measurements

    Parameters
    ----------
    amp: string
        amp series - box number - and channel. Ex: 300-01
    channel: int
        channel id
    path: string
        directory where amp responses are located. Default is "HardwareResponses/"
    """

    # load null measurement
    amp_gain_db = np.loadtxt(os.path.join(path, 'LOGMAG_NULL.CSV'), skiprows=3, delimiter=',')
    ph = os.path.join(path, 'PHASE_NULL.CSV')
    amp_phase_discrete = np.loadtxt(ph, skiprows=3, delimiter=',')

    # Convert to GHz and add 40dB for attenuation in measurement circuit
    amp_gain_db[:, 0] *= units.Hz
    amp_phase_discrete[:, 0] *= units.Hz
    amp_phase_discrete[:, 1] *= units.deg
    if(not np.allclose(amp_gain_db[:, 0], amp_phase_discrete[:, 0])):
        raise ValueError("frequencies of gain and phase measurement are not equal for NULL measurement")
    amp_gain_db[:, 1] += 40  # 300 series amps
    gain_lin = 10 ** (amp_gain_db[:, 1] / 20.)
    response_null = gain_lin * np.exp(1j * amp_phase_discrete[:, 1])


    # read in individual amp measurement
    path = os.path.join(path, 'Box' + amp)
    ch_name = '_CH{:d}.CSV'.format(channel)
    amp_gain_db = np.loadtxt(os.path.join(path, 'LOGMAG' + ch_name), skiprows=3, delimiter=',')
    ph = os.path.join(path, 'PHASE' + ch_name)
    amp_phase_discrete = np.loadtxt(ph, skiprows=3, delimiter=',')

    # Convert to GHz and add 40dB for attenuation in measurement circuit
    amp_gain_db[:, 0] *= units.Hz
    amp_phase_discrete[:, 0] *= units.Hz
    amp_phase_discrete[:, 1] *= units.deg
    ff = amp_gain_db[:, 0]
    if(not np.allclose(amp_gain_db[:, 0], amp_phase_discrete[:, 0])):
        raise ValueError("frequencies of gain and phase measurement are not equal for {}-{:02d}".format(amp, channel))

    amp_gain_db[:, 1] += 40  # 300 series amps

    gain_lin = 10 ** (amp_gain_db[:, 1] / 20.)
    r = gain_lin * np.exp(1j * amp_phase_discrete[:, 1])

    response = {}
    response['freqs'] = ff
    response['response'] = r / response_null
    amplifier_response = {}
    amp_name = '{}-{:02d}'.format(amp, channel)
    amplifier_response[amp_name] = response

    # checkGroupDelay(get_amp_gain(ff)*get_amp_phase(ff),ff,ch_name,path)
    save_preprocessed_Amps(amplifier_response, amp_name)


for i in range(8):
    preprocess_300Amp('300-03', i, path='HardwareResponses/300SeriesAmpBoxes')
