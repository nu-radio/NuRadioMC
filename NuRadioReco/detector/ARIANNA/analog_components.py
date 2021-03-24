import numpy as np
from scipy.interpolate import interp1d
import os
from radiotools import helper as hp
from NuRadioReco.utilities import units, io_utilities
import logging

logger = logging.getLogger('analog_components')


def load_amplifier_response(amp_type='100', path=os.path.dirname(os.path.realpath(__file__))):
    """
    Read out amplifier gain and phase. Currently only examples have been implemented.
    Needs a better structure in the future, possibly with database.
    """
    amp_response = {}
    if amp_type == '100':
        amp_gain_discrete = np.loadtxt(os.path.join(path, 'HardwareResponses/Amp109_SHP100SLP1000_3L3_60dB.csv'),
                                       skiprows=44, delimiter=',')
        ph = os.path.join(path, 'HardwareResponses/AMP109_SHP100SLP1000_3L3_PHASE.CSV')
        amp_phase_discrete = np.loadtxt(ph, skiprows=3, delimiter=',')
    elif amp_type == '200':
        amp_gain_discrete = np.loadtxt(os.path.join(path, 'HardwareResponses/amp_200_logmag.csv'),
                                       skiprows=3, delimiter=',')
        ph = os.path.join(path, 'HardwareResponses/amp_200_phase.csv')
        amp_phase_discrete = np.loadtxt(ph, skiprows=3, delimiter=',')
    elif amp_type == '300':
        amp_gain_discrete = np.loadtxt(os.path.join(path, 'HardwareResponses/amp_300_gain.csv'),
                                       skiprows=3, delimiter=',')
        ph = os.path.join(path, 'HardwareResponses/amp_300_phase.csv')
        amp_phase_discrete = np.loadtxt(ph, skiprows=3, delimiter=',')
    else:
        logger.error("Amp type not recognized")
        return amp_response

    # Convert to GHz and add 60dB/40dB for attenuation in measurement circuit
    amp_gain_discrete[:, 0] *= units.Hz

    if amp_type == '300':
        amp_gain_discrete[:, 1] += 40
    elif amp_type == '100':
        amp_gain_discrete[:, 1] += 60
    elif amp_type == '200':
        amp_gain_discrete[:, 1] += 60

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

    amp_response['gain'] = get_amp_gain
    amp_response['phase'] = get_amp_phase

    return amp_response


def load_amp_measurement(amp_measurement):
    """
    load individual amp measurement from file and buffer interpolation function
    """
    filename = os.path.join(os.path.dirname(__file__), 'HardwareResponses/', amp_measurement + ".pkl")
    data = io_utilities.read_pickle(filename, encoding='latin1')
    if amp_measurement not in data:
        raise AttributeError("can't find amp measurement {}".format(amp_measurement))
    ff = data[amp_measurement]['freqs']
    response = data[amp_measurement]['response']
    gain = np.abs(response)
    phase = np.unwrap(np.angle(response))
    amp_phase_f = interp1d(ff, phase, bounds_error=False,
                           fill_value=0)  # all requests outside of measurement range are set to 0
    amp_gain_f = interp1d(ff, gain, bounds_error=False,
                          fill_value=1)  # all requests outside of measurement range are set to 1

    def get_response(freqs):
        return amp_gain_f(freqs) * np.exp(1j * amp_phase_f(freqs))

    amp_measurements[amp_measurement] = get_response


# amp responses do not occupy a lot of memory, pre load all responses
amplifier_response = {}
for amplifier_type in ['100', '200', '300']:
    amplifier_response[amplifier_type] = load_amplifier_response(amplifier_type)

amp_measurements = {}  # buffer for amp measurements


def get_amplifier_response(ff, amp_type, amp_measurement=None):
    if amp_measurement is not None:
        if amp_measurement not in amp_measurements:
            load_amp_measurement(amp_measurement)
        return amp_measurements[amp_measurement](ff)
    elif amp_type in amplifier_response.keys():
        return amplifier_response[amp_type]['gain'](ff) * amplifier_response[amp_type]['phase'](ff)
    else:
        logger.error("Amplifier response for type {} not implemented, returning None".format(amp_type))
        return None


def get_cable_response_parametrized(frequencies, cable_type, cable_length):
    if cable_type == "LMR_400":

        def attn_db_per_100ft(f):  # from LMR-400 spec sheet
            return 0.122290 * (f / units.MHz) ** 0.5 + 0.000260 * f / units.MHz  # https://www.timesmicrowave.com/DataSheets/CableProducts/LMR-400.pdf

        logger.debug("{} {} {}".format(cable_type, cable_length, type(cable_length)))
        attn = attn_db_per_100ft(frequencies) / (100 * units.feet) * cable_length
        attn += 0.01  # dB connector loss
        return 1. / hp.dB_to_linear(attn) ** 0.5
    elif cable_type == "LMR_240":

        def attn_db_per_100ft(f):  # from LMR-400 spec sheet
            return 0.242080 * (f / units.MHz) ** 0.5 + 0.000330 * f / units.MHz  # https://www.timesmicrowave.com/DataSheets/CableProducts/LMR-240.pdf

        logger.debug("{} {} {}".format(cable_type, cable_length, type(cable_length)))
        attn = attn_db_per_100ft(frequencies) / (100 * units.feet) * cable_length
        attn += 0.01  # dB connector loss
        return 1. / hp.dB_to_linear(attn) ** 0.5
    else:
        logger.error("cable type {} not defined".format(cable_type))
        raise NotImplementedError


def get_cable_response(frequencies, path=os.path.dirname(os.path.realpath(__file__))):
    """
    Read out cable induced loss and phase. From standard 4 channel station.
    """

    cable_discrete = np.loadtxt(os.path.join(path, 'HardwareResponses/CableAntennuation_James2016.csv'), skiprows=1,
                                delimiter=',')

    max_frequency = 5000. * units.MHz
    if np.max(frequencies) > max_frequency:
        max_frequency = np.max(frequencies)

    # Convert to GHz
    cable_discrete[:, 0] *= units.Hz
    cable_discrete[0, 0] = 0.
    cable_discrete[-1, 0] = max_frequency

    cable_amp_db_f = interp1d(cable_discrete[:, 0], cable_discrete[:, 1])
    cable_amp_db = cable_amp_db_f(frequencies)
    cable_amp = 10 ** (cable_amp_db / 20.)

    cable_phase_f = interp1d(cable_discrete[:, 0], np.unwrap(np.deg2rad(cable_discrete[:, 2])))
    cable_phase = cable_phase_f(frequencies)
    cable_phase = np.exp(1j * cable_phase)

    return cable_amp * cable_phase


def get_available_amplifiers():
    return ['100', '200', '300']
