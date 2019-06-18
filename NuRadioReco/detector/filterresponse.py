import numpy as np
from NuRadioReco.utilities import units
from radiotools import helper as hp
from scipy import interpolate as intp
from scipy import integrate as int
import os
import glob


def get_filter_response_mini_circuits(frequencies, filter_name):
    """
    Simple minicircuits filters.
    Check filter directory for available filters.

    """
    # parse input file and convert to default units
    directory = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(directory, 'filter/{}_S2P'.format(filter_name), '{}_Minus40degC.S2P'.format(filter_name))
    if not os.path.exists(filename):
        filename = os.path.join(directory, 'filter/{}_S2P'.format(filter_name), '{}__UNIT_1.s2p'.format(filter_name))
        if not os.path.exists(filename):
            filename = glob.glob(os.path.join(directory, 'filter/{}_S2P'.format(filter_name), '{}*'.format(filter_name)))[0]
    data = np.loadtxt(filename, comments=['#', '!'])
    ff = data.T[0] * units.MHz
    S21 = 10**(data.T[3] / 20.) * np.exp(1j * data.T[4] * units.deg)

    get_S21 = intp.interp1d(ff, S21)

    response = np.zeros_like(frequencies, dtype=np.complex)
    mask = (frequencies > ff.min()) & (frequencies < ff.max())
    response[mask] = get_S21(frequencies[mask])
    return response


def get_filter_response_mini_circuits2(frequencies, filter_name):
    """
    Simple minicircuits filters.
    Check filter directory for available filters.
    Parse input file and convert to default units

    """

    directory = os.path.dirname(os.path.abspath(__file__))
    ff, insertion_loss, return_loss, ff2, group_delay = np.loadtxt(
        os.path.join(directory, 'filter/{}.dat'.format(filter_name)), unpack=True)
    ff *= units.MHz
    ff2 *= units.MHz
    group_delay *= units.ns
    insertion_loss = 10**(-insertion_loss / 20.)
    return_loss = 10**(-return_loss / 20.)

    get_insertion_loss = intp.interp1d(ff, insertion_loss)

    get_group_delay = intp.interp1d(ff2, group_delay)
    df = 1 * units.MHz
    fff2 = np.linspace(ff2.min(), ff2.max(), np.int(np.ceil((ff2.max() - ff2.min()) / (1 * units.MHz))))
    phase2 = -2 * np.pi * np.cumsum(get_group_delay(fff2) * df)
    get_phase = intp.interp1d(fff2, phase2)

    response = np.zeros_like(frequencies, dtype=np.complex)
    mask = (frequencies > max(ff.min(), ff2.min())) & (frequencies < min(ff.max(), ff2.max()))
    response[mask] = get_insertion_loss(frequencies[mask]) * np.exp(1j * get_phase(frequencies[mask]))
    return response


def get_filter_response(frequencies, filter_name):
    """
    Get measured filter responses.
    """
    directory = os.path.dirname(os.path.abspath(__file__))
    if(filter_name =='NTU+cheb'):
        ff, mag, phase = np.loadtxt(os.path.join(directory, 'filter/NTU+cheb_filter_mag_phase.txt'), unpack=True)
        get_phase = intp.interp1d(ff, np.unwrap(phase))
        get_insertion_loss = intp.interp1d(ff, mag)
        ff2 = ff
    else:
        # parse input file and convert to default units
        ff2, phase, t = np.loadtxt(os.path.join(
            directory, 'filter/measurement/{}_PHASE.CSV'.format(filter_name)), unpack=True, skiprows=3, delimiter=',')
        ff, insertion_loss, t = np.loadtxt(os.path.join(
            directory, 'filter/measurement/{}_LINMAG.CSV'.format(filter_name)), unpack=True, skiprows=3, delimiter=',')
        ff *= units.Hz
        ff2 *= units.Hz
        phase *= units.deg

        get_phase = intp.interp1d(ff2, np.unwrap(phase))
        get_insertion_loss = intp.interp1d(ff, insertion_loss)

    response = np.zeros_like(frequencies, dtype=np.complex)
    mask = (frequencies > max(ff.min(), ff2.min())) & (frequencies < min(ff.max(), ff2.max()))
    response[mask] = get_insertion_loss(frequencies[mask]) * np.exp(1j * get_phase(frequencies[mask]))
    return response


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    ff = np.linspace(10 * units.MHz, 500 * units.MHz, 100)
    df = ff[1] - ff[0]
    fig, (ax, ax2) = plt.subplots(1, 2)
    for name in ["SHP-48+", "SHP-20+", "SHP-25+", "SXHP-48+", "ZFHP-1R2-S+", "ZX75HP-44+"]:
        response1 = get_filter_response_mini_circuits(ff, name)
        ax.plot(ff / units.MHz, np.abs(response1), label=name)
        group_delay = -1./2/np.pi * np.diff(np.unwrap(np.angle(response1))) / df
    #     ax2.plot(ff / units.MHz, np.unwrap(np.angle(response1)) / units.deg)
        ax2.plot((ff[1:] + ff[:-1]) * 0.5 / units.MHz, group_delay/units.ns, label=name)
    ax.legend()
    ax2.legend()
    fig.tight_layout()
    plt.show()
    ff = np.linspace(200 * units.MHz, 800 * units.MHz, 100)
    fig, (ax, ax2) = plt.subplots(1, 2)
    for name in ["ZX75LP-470+", "ZFLP-450+", "SBLP-467+"]:
        response1 = get_filter_response_mini_circuits(ff, name)
        ax.plot(ff / units.MHz, np.abs(response1), label=name)
        group_delay = -1./2/np.pi * np.diff(np.unwrap(np.angle(response1))) / df
    #     ax2.plot(ff / units.MHz, np.unwrap(np.angle(response1)) / units.deg)
        ax2.plot((ff[1:] + ff[:-1]) * 0.5 / units.MHz, group_delay/units.ns, label=name)
    ax.legend()
    ax2.legend()
    fig.tight_layout()
    plt.show()

    response1 = get_filter_response_mini_circuits(ff, "SHP-100+")
    response2 = get_filter_response(ff, "SHP-100+")

    fig, (ax, ax2) = plt.subplots(1, 2)
    phase1 = np.unwrap(np.angle(response1))
    phase1 = phase1 / phase1.min()
    phase2 = np.unwrap(np.angle(response2))
    phase2 = phase2 / phase2.min()
    ax.plot(ff / units.MHz, phase1, label='spec sheet')
    ax.plot(ff / units.MHz, phase2, label='measurement')

    ax2.plot(ff[1:] / units.MHz, np.diff(np.unwrap(np.angle(response1))), label='spec sheet')
    ax2.plot(ff[1:] / units.MHz, np.diff(np.unwrap(np.angle(response2))), label='measurement')
    ax.legend()
    plt.show()
