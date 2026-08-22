from NuRadioReco.utilities import units, io_utilities
from radiotools import helper as hp

from scipy import constants
from scipy.signal.windows import hann
from time import time
import numpy as np
import logging
import pickle
import cmath
import scipy
import json
import csv
import os

logger = logging.getLogger('NuRadioReco.antennapattern')
path_to_antennamodels = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AntennaModels')


def interpolate_linear(x, x0, x1, y0, y1, interpolation_method='complex'):
    """
    Linearly interpolate between two (arrays of) complex numbers

    All arguments may be scalars or arrays as long as they broadcast against each other.
    Where the two data points coincide (``x0 == x1``) ``y0`` is returned.

    Parameters
    ----------
    x: float or array of floats
        the requested position(s)
    x0, y0: float or complex float (or arrays thereof)
        the first data point
    x1, y1: float or complex float (or arrays thereof)
        the second data point
    interpolation_method: string
        specifies if interpolation is in

        * complex (default) i.e. real and imaginary part
        * magnitude and phase

    Returns
    -------
    y: complex float or array of complex floats
        the interpolated value(s)
    """
    delta = np.asarray(x1) - np.asarray(x0)
    coincident = delta == 0
    weight = np.where(coincident, 0., (np.asarray(x) - np.asarray(x0)) / np.where(coincident, 1., delta))

    if interpolation_method == 'complex':
        return y0 + (y1 - y0) * weight

    if interpolation_method == 'magphase':  # interpolate magnitude and phase
        # unwrap along the axis connecting the two data points, i.e. do not confuse
        # a phase wrap between y0 and y1 with the phase evolution within either of them
        phase0, phase1 = np.unwrap([np.angle(y0), np.angle(y1)], axis=0)
        mag = np.abs(y0) + (np.abs(y1) - np.abs(y0)) * weight
        return mag * np.exp(1j * (phase0 + (phase1 - phase0) * weight))

    logger.error("interpolation mode {} not implemented".format(interpolation_method))
    raise NotImplementedError("interpolation mode {} not implemented".format(interpolation_method))


# `interpolate_linear` handles arrays, the two functions used to be separate
interpolate_linear_vectorized = interpolate_linear


def get_interpolation_weight(x, x0, x1):
    """
    Fraction of the way from ``x0`` to ``x1`` at which ``x`` lies, 0 for x0 == x1
    """
    if x0 == x1:
        return 0.

    return (x - x0) / (x1 - x0)


def is_equidistant(nodes, rtol=1e-4):
    """
    Check whether the grid nodes are equally spaced (within ``rtol`` of the spacing)
    """
    if len(nodes) < 3:
        return True

    spacing = np.diff(nodes)
    return bool(np.ptp(spacing) < rtol * np.abs(np.mean(spacing)))


def get_bracketing_indices(x, nodes, equidistant=False):
    """
    Find the indices of the two grid nodes bracketing the requested position(s)

    The nodes are only required to be sorted, not equally spaced. Positions outside
    the node range are clamped to the outermost interval (i.e. they are extrapolated);
    callers are expected to handle out-of-range values themselves if that is not wanted.

    Parameters
    ----------
    x: float or array of floats
        the requested position(s)
    nodes: array of floats
        the (sorted) grid nodes
    equidistant: bool (default: False)
        if True the indices are calculated from the grid boundaries instead of looked up.
        This is much faster for large ``x`` but only correct for equally spaced nodes
        (cf. `is_equidistant`).

    Returns
    -------
    i_lower, i_upper: array of ints
        indices of the nodes below and above ``x``. They are identical if the grid
        has only a single node.
    """
    n = len(nodes)
    if n == 1:
        zero = np.zeros_like(x, dtype=int)
        return zero, zero

    if equidistant:
        i_upper = np.ceil((x - nodes[0]) / (nodes[-1] - nodes[0]) * (n - 1))
    else:
        i_upper = np.searchsorted(nodes, x, side='left')

    i_upper = np.array(np.clip(i_upper, 1, n - 1), dtype=int)
    return i_upper - 1, i_upper


def get_group_delay(vector_effective_length, df):
    """
    helper function to calculate the group delay from the vector effecitve length

    Parameters
    ----------
    vector_effective_length: complex float
        the vector effective length of an antenna
    df: float
        the size of a frequency bin

    Returns
    -------
    dt: float
        the group delay


    """
    return -np.diff(np.unwrap(np.angle(vector_effective_length))) / df / units.ns / 2 / np.pi


def parse_RNOG_XFDTD_file(path_gain, path_phases, encoding = None):
    """"
    reads in XFDTD data

    Parameters
    ----------
    path_gain: string
        path to gain file
    path_phases:
        path to phases file

    Returns
    -------
    all paramters of the file as numpy arrays
    """""

    with open(path_gain, 'r', encoding = encoding) as fin:
        ff = []
        phis = []
        thetas = []
        gain_theta = []
        gain_phi = []
        csv_reader = csv.reader(fin, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if 1:  # (line_count % 2) == 0:
                if line_count != 0:
                    ff.append(float(row[0]))
                    thetas.append(float(row[1]))
                    phis.append(float(row[2]))
                    gain_phi.append(float(row[3]))
                    gain_theta.append(float(row[4]))

            line_count += 1

    with open(path_phases, 'r', encoding = encoding) as fin:
        phase_phi = []
        phase_theta = []
        csv_reader = csv.reader(fin, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if 1:  # (line_count % 2) == 0:
                if line_count != 0:
                    complex = float(row[3]) + 1j * float(row[4])
                    phase_phi.append(cmath.phase(complex))
                    complex = float(row[5]) + 1j * float(row[6])
                    phase_theta.append(cmath.phase(complex))

            line_count += 1

    return np.array(ff), np.array(phis), np.array(thetas), np.array(gain_phi), np.array(gain_theta), np.array(phase_phi), np.array(phase_theta)


def preprocess_RNOG_XFDTD(path_gain, path_phases, outputfilename, n_index=1.74, encoding = None):
    """"
    Preprocess an antenna pattern in XFDTD file format. The vector effective length is calculated and the output is saved to the NuRadioReco pickle format.

    This conversion function ASSUMES THAT THE XFDTD SIMULATION IS DONE IN AIR! HERE WE DO A FIRST ORDER RESCALING
    TO A DIFFERENT INDEX OF REFRACTION by just rescaling the frequencies by f -> f/n.

    Parameters
    ----------
    path_gain: string
        path to gain file
    path_phases: string
        path to phases file
    outputfilename: string
        path to outputfilename
    n_index: float
        refractive index for requested antenna file. The method assumes that simulations are done in air (n = 1)
    """

    ff, phi, theta, gain_phi, gain_theta, phase_phi, phase_theta = parse_RNOG_XFDTD_file(path_gain, path_phases, encoding = encoding)
    c = constants.c * units.m / units.s
    Z_0 = 119.9169 * np.pi # free space impedance

    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)

    wavelength = c / np.array(ff)

    H_theta = wavelength * (50 / (4 * np.pi * Z_0)) ** 0.5 * gain_theta ** 0.5 * np.exp(1j * phase_theta)
    H_phi = wavelength * (50 / (4 * np.pi * Z_0)) ** 0.5 * gain_phi ** 0.5 * np.exp(1j * phase_phi)

    # orientation and rotation angles of the antenna for which the pattern is given
    zen_boresight = 0
    azi_boresight = 0
    zen_ori = 0.5 * np.pi
    azi_ori = 0

    index = np.lexsort((theta, phi, ff))
    ff = np.array(ff)[index]
    phi = phi[index]
    theta = theta[index]
    H_phi = np.array(H_phi)[index]
    H_theta = np.array(H_theta)[index]

    # rescale frequencies from air to medium with `n_index`
    ff = ff / n_index

    with open(outputfilename, 'wb') as fout:
        pickle.dump([zen_boresight, azi_boresight, zen_ori, azi_ori, ff, theta, phi, H_phi, H_theta], fout, protocol=2)


def parse_WIPLD_file(ad1, ra1, orientation, gen_num=1, s_parameters=None):
    """
    reads in WIPLD data

    Parameters
    ----------
    ad1: string
        path to ad1 file
    ra1: string
        path to radiation pattern file
    orientation: string
        path to orientation file
    gen_num: int
        which antenna (one or two) to pull from
    s_parameters: list of 2 ints
        determines which s-parametr to extract (ex: [1,2] extracts S_12 parameter).

    Returns
    -------
    all parameters of the files
    """
    if s_parameters is None:
        s_parameters = [1, 1]
    boresight, tines = np.loadtxt(orientation, delimiter=',')
    orientation_theta, orientation_phi = hp.cartesian_to_spherical(*boresight)
    rotation_theta, rotation_phi = hp.cartesian_to_spherical(*tines)

    ad1_data = np.loadtxt(ad1, comments='>')
    S_1 = ad1_data[:, 1]
    S_2 = ad1_data[:, 2]
    mask = (S_1 == s_parameters[0]) & (S_2 == s_parameters[1])
    ff = ad1_data[:, 0][mask] * units.GHz
    Re_Z = ad1_data[:, 5][mask] * units.ohm
    Im_Z = ad1_data[:, 6][mask] * units.ohm
    Z = Re_Z + 1j * Im_Z

    Re_S = ad1_data[:, 7][mask]
    Im_S = ad1_data[:, 8][mask]
    S = Re_S + 1j * Im_S
    with open(ra1, 'r') as fin:
        ff2 = []
        phis = []
        thetas = []
        Ephis = []
        Ethetas = []
        gains = []
        f = None
        skip = False
        for line in fin.readlines():
            if line.strip().startswith('>'):
                skip = False
                if int(line.split()[3]) != gen_num:
                    skip = True
                else:
                    logger.debug(line.split())
                f = float(line.split()[4])
            else:
                if skip:
                    continue
                ff2.append(f * units.GHz)
                phi, theta, ReEphi, ImEphi, ReEtheta, ImEtheta, gain, gaindb = line.split()
                phis.append(float(phi))
                thetas.append(float(theta))
                Ephis.append(float(ReEphi) + 1j * float(ImEphi))
                Ethetas.append(float(ReEtheta) + 1j * float(ImEtheta))
                gains.append(float(gain))

        if not np.array_equal(ff, np.unique(np.array(ff2))):
            logger.error("error in parsing WIPLD simulation, frequencies of ad1 and ra1 files do not match!")
            return None
        logger.debug(np.unique(np.array(phis)))
        logger.debug(np.unique(np.array(thetas)))
        return orientation_theta, orientation_phi, rotation_theta, rotation_phi, ff, Z, S, np.array(ff2), np.deg2rad(
            np.array(phis)), np.deg2rad(np.array(thetas)), np.array(Ephis), np.array(Ethetas), np.array(gains)


def preprocess_WIPLD_old(path, gen_num=1, s_parameters=None):
    """
    preprocesses WIPLD file

    this function implements the older insufficient calculation of the vector effective length. This VEL only
    relates the incident electric field to the open circuit voltage and not the voltage in a 50 Ohm system.

    Parameters
    ----------
    path: string
        path to folder containing ad1, ra1, and orientation files.
    gen_num: int
        which antenna (one or two) to pull from
    s_parameters: list of 2 ints
        determines which s-parametr to extract (ex: [1,2] extracts S_12 parameter).

    Returns
    -------
    orientation theta: float
        orientation of the antenna, as a zenith angle (0deg is the zenith, 180deg is straight down); for LPDA: outward along boresight; for dipoles: upward along axis of azimuthal symmetry
    orientation phi: float
        orientation of the antenna, as an azimuth angle (counting from East counterclockwise); for LPDA: outward along boresight; for dipoles: upward along axis of azimuthal symmetry
    rotation theta: float
        rotation of the antenna, is perpendicular to 'orientation', for LPDAs: vector perpendicular to the plane containing the the tines
    rotation phi: float
        rotation of the antenna, is perpendicular to 'orientation', for LPDAs: vector perpendicular to the plane containing the the tines
    ff2: array of floats
        array of frequencies
    theta: float
        zenith angle of inicdent electric field
    phi: float
        azimuth angle of incident electric field
    H_phi: float
        the complex realized vector effective length of the ePhi polarization component
    H_theta: float
        the complex realized vector effective length of the eTheta polarization component
    """
    if s_parameters is None:
        s_parameters = [1, 1]
    from scipy.interpolate import interp1d
    c = constants.c * units.m / units.s
    Z_0 = 119.9169 * np.pi * units.ohm
    split = os.path.split(os.path.dirname(path))
    name = split[1]
    path = split[0]

    orientation_theta, orientation_phi, rotation_theta, rotation_phi, ff, Z, S, ff2, phi, theta, Iphi, Itheta, gains = parse_WIPLD_file(
        os.path.join(path, name, '{}.ad1'.format(name)),
        os.path.join(path, name, '{}.ra1'.format(name)),
        os.path.join(path, name, '{}.orientation'.format(name)),
        gen_num=gen_num, s_parameters=s_parameters)

    theta = 0.5 * np.pi - theta  # 90deg - theta because in WIPL D the theta angle is defined differently

    # sort with increasing frequency, increasing phi, and increasing theta
    index = np.lexsort((theta, phi, ff2))
    ff2 = ff2[index]
    phi = phi[index]
    theta = theta[index]
    Iphi = Iphi[index]
    Itheta = Itheta[index]

    get_Z = interp1d(ff, Z, kind='nearest')
    wavelength = c / ff2
    H_phi = (2 * wavelength * get_Z(ff2) * Iphi) / Z_0 / 1j
    # need a minus sign in H_theta because eTheta points in the opposite direction
    # in NuRadio compared to WIPL-D
    H_theta = -(2 * wavelength * get_Z(ff2) * Itheta) / Z_0 / 1j

    return orientation_theta, orientation_phi, rotation_theta, rotation_phi, ff2, theta, phi, H_phi, H_theta


#     H = wavelength * (np.real(get_Z(ff2)) / (np.pi * Z_0)) ** 0.5 * gains ** 0.5


def save_preprocessed_WIPLD_old(path):
    """
    saves preprocessed WIPLD files to a pickle file

    Parameters
    ----------
    path: string
        path to folder containing ad1, ra1, and orientation files.
    """
    orientation_theta, orientation_phi, rotation_theta, rotation_phi, ff2, theta, phi, H_phi, H_theta = preprocess_WIPLD_old(
        path)
    split = os.path.split(os.path.dirname(path))
    name = split[1]
    path = split[0]
    output_filename = '{}.pkl'.format(os.path.join(path, name, name))
    with open(output_filename, 'wb') as fout:
        logger.info('saving output to {}'.format(output_filename))
        pickle.dump([orientation_theta, orientation_phi, rotation_theta, rotation_phi, ff2, theta, phi, H_phi, H_theta],
                    fout, protocol=4)


def preprocess_WIPLD(path, gen_num=1, s_parameters=None):
    """
    preprocesses WIPLD file

    this function implements the older insufficient calculation of the vector effective length. This VEL only
    relates the incident electric field to the open circuit voltage and not the voltage in a 50 Ohm system.

    Parameters
    ----------
    path: string
        path to folder containing ad1, ra1, and orientation files.
    gen_num: int
        which antenna (one or two) to pull from
    s_parameters: list of 2 ints
        determines which s-parametr to extract (ex: [1,2] extracts S_12 parameter).

    Returns
    -------
    orientation theta: float
        orientation of the antenna, as a zenith angle (0deg is the zenith, 180deg is straight down); for LPDA: outward along boresight; for dipoles: upward along axis of azimuthal symmetry
    orientation phi: float
        orientation of the antenna, as an azimuth angle (counting from East counterclockwise); for LPDA: outward along boresight; for dipoles: upward along axis of azimuthal symmetry
    rotation theta: float
        rotation of the antenna, is perpendicular to 'orientation', for LPDAs: vector perpendicular to the plane containing the the tines
    rotation phi: float
        rotation of the antenna, is perpendicular to 'orientation', for LPDAs: vector perpendicular to the plane containing the the tines
    ff2: array of floats
        array of frequencies
    theta: float
        zenith angle of inicdent electric field
    phi: float
        azimuth angle of incident electric field
    H_phi: float
        the complex realized vector effective length of the ePhi polarization component
    H_theta: float
        the complex realized vector effective length of the eTheta polarization component
    """
    if s_parameters is None:
        s_parameters = [1, 1]
    from scipy.interpolate import interp1d
    c = constants.c * units.m / units.s
    Z_0 = 119.9169 * np.pi * units.ohm
    split = os.path.split(os.path.dirname(path))
    name = split[1]
    path = split[0]

    orientation_theta, orientation_phi, rotation_theta, rotation_phi, ff, Z, S, ff2, phi, theta, Iphi, Itheta, gains = parse_WIPLD_file(
        os.path.join(path, name, '{}.ad1'.format(name)),
        os.path.join(path, name, '{}.ra1'.format(name)),
        os.path.join(path, name, '{}.orientation'.format(name)),
        gen_num=gen_num, s_parameters=s_parameters)

    theta = 0.5 * np.pi - theta  # 90deg - theta because in WIPL D the theta angle is defined differently

    # sort with increasing frequency, increasing phi, and increasing theta
    index = np.lexsort((theta, phi, ff2))
    ff2 = ff2[index]
    phi = phi[index]
    theta = theta[index]
    Iphi = Iphi[index]
    Itheta = Itheta[index]

    #     get_Z = interp1d(ff, Z, kind='nearest')
    get_S = interp1d(ff, S, kind='nearest')
    wavelength = c / ff2
    V = 1 * units.V
    Z_L = 50 * units.ohm
    H_phi = wavelength * (1 + get_S(ff2)) * Iphi * Z_L / Z_0 / 1j / V
    # need a minus sign in H_theta because eTheta points in the opposite direction
    # in NuRadio compared to WIPL-D
    H_theta = -wavelength * (1 + get_S(ff2)) * Itheta * Z_L / Z_0 / 1j / V

    #     H = wavelength * (np.real(get_Z(ff2)) / (np.pi * Z_0)) ** 0.5 * gains ** 0.5
    return orientation_theta, orientation_phi, rotation_theta, rotation_phi, ff2, theta, phi, H_phi, H_theta


def save_preprocessed_WIPLD(path):
    """
    saves preprocessed WIPLD files to a pickle file

    Parameters
    ----------
    path: string
        path to folder containing ad1, ra1, and orientation files.
    """
    orientation_theta, orientation_phi, rotation_theta, rotation_phi, ff2, theta, phi, H_phi, H_theta = preprocess_WIPLD(
        path)
    split = os.path.split(os.path.dirname(path))
    name = split[1]
    path = split[0]
    output_filename = '{}.pkl'.format(os.path.join(path, name, name))
    with open(output_filename, 'wb') as fout:
        logger.info('saving output to {}'.format(output_filename))
        pickle.dump([orientation_theta, orientation_phi, rotation_theta, rotation_phi, ff2, theta, phi, H_phi, H_theta],
                    fout, protocol=4)


def save_preprocessed_WIPLD_forARA(path):
    """
    this function saves the realized gain in an ARASim readable format

    Parameters
    ----------
    path: string
        path to folder containing ad1, ra1, and orientation files.
    """
    from scipy.interpolate import interp1d
    c = constants.c * units.m / units.s
    Z_0 = 119.9169 * np.pi * units.ohm
    split = os.path.split(os.path.dirname(path))
    name = split[1]
    path = split[0]

    orientation_theta, orientation_phi, rotation_theta, rotation_phi, ff, Z, S, ff2, phi, theta, Iphi, Itheta, gains = parse_WIPLD_file(
        os.path.join(path, name, '{}.ad1'.format(name)),
        os.path.join(path, name, '{}.ra1'.format(name)),
        os.path.join(path, name, '{}.orientation'.format(name)))

    theta = 0.5 * np.pi - theta  # 90deg - theta because in WIPL D the theta angle is defined differently

    # sort with increasing frequency, increasing phi, and increasing theta
    index = np.lexsort((theta, phi, ff2))
    ff2 = ff2[index]
    phi = phi[index]
    theta = theta[index]
    Iphi = Iphi[index]
    Itheta = Itheta[index]

    wavelength = c / ff2
    V = 1 * units.V
    Z_L = 50 * units.ohm

    get_S = interp1d(ff, S, kind='nearest')
    Gr = gains * (1 - np.abs(get_S(ff2)) ** 2)
    H_phi = wavelength * (1 + get_S(ff2)) * Iphi * Z_L / Z_0 / 1j / V
    # need a minus sign in H_theta because eTheta points in the opposite direction
    # in NuRadio compared to WIPL-D
    H_theta = -wavelength * (1 + get_S(ff2)) * Itheta * Z_L / Z_0 / 1j / V

    output_filename = '{}.ara'.format(os.path.join(path, name, name))
    with open(output_filename, 'w') as fout:
        for f in sorted(np.unique(ff2)):
            fout.write("freq : {} MHz\n".format(f / units.MHz))
            fout.write("SWR : ???\n")
            fout.write("Theta   Phi      Gain(dB)          Gain          Phase(deg)\n")
            mask = ff2 == f
            for i in range(np.sum(mask)):
                fout.write("{:.4f} {:.4f} {:.4g} {:.4g} {:.2f} {:.2f}\n".format(theta[mask][i] / units.deg,
                                                                                phi[mask][i] / units.deg,
                                                                                0,
                                                                                Gr[mask][i],
                                                                                np.angle(H_theta[mask][i]) / units.deg,
                                                                                np.angle(H_phi[mask][i]) / units.deg))

def get_pickle_antenna_response(path, return_verified=False):
    """
    Opens and return the pickle file containing the preprocessed e.g. WIPL-D antenna simulation in NuRadioReco conventions.

    If the pickle file is not present on the local file system, or if the file is outdated (verified via a sha1 hash sum),
    the file will be downloaded from a central data server.

    Parameters
    ----------
    path: string
        the path to the pickle file
    return_verified: bool (default: False)
        if True, the function will return a boolean indicating whether the file was verified/downloaded or not.

    Returns
    -------
    res: 9 lists
        list containing the following elements:

        * orientation_theta: float
            orientation of the antenna, as a zenith angle (0deg is the zenith, 180deg is straight down); for LPDA: outward along boresight; for dipoles: upward along axis of azimuthal symmetry
        * orientation_phi: float
            orientation of the antenna, as an azimuth angle (counting from East counterclockwise); for LPDA: outward along boresight; for dipoles: upward along axis of azimuthal symmetry
        * rotation_theta: float
            rotation of the antenna, is perpendicular to 'orientation', for LPDAs: vector perpendicular to the plane containing the the tines
        * rotation_phi: float
            rotation of the antenna, is perpendicular to 'orientation', for LPDAs: vector perpendicular to the plane containing the the tines
        * ff: array of floats
            array of frequencies
        * thetas: array of floats
            zenith angle of inicdent electric field
        * phis: array of floats
            azimuth angle of incident electric field
        * H_phi: array of floats
            the complex realized vector effective length of the ePhi polarization component
        * H_theta: array of floats
            the complex realized vector effective length of the eTheta polarization component

    verified: bool
        boolean indicating whether the file was verified/downloaded or not (only returned if ``return_verified == True``)
    """

    download_file = False
    verified = False

    # check if gziped pickle file already exists
    if not os.path.exists(path):
        logger.status("antenna pattern {} does not exist, file will be downloaded".format(path))
        download_file = True

    if os.path.exists(path):
        BUF_SIZE = 65536 * 2 ** 4  # lets read stuff in 64kb chunks!
        import hashlib
        import json
        sha1 = hashlib.sha1()
        with open(path, 'rb') as f:
            while True:
                data = f.read(BUF_SIZE)
                if not data:
                    break
                sha1.update(data)

        antenna_directory = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(antenna_directory, 'antenna_models_hash.json'), 'r') as fin:
            antenna_hashs = json.load(fin)
            logger.info('search for {}'.format(os.path.basename(path)))
            if os.path.basename(path) in antenna_hashs.keys():
                verified = True  # either hash is correct or the file is downloaded
                if sha1.hexdigest() != antenna_hashs[os.path.basename(path)]:
                    logger.status("antenna model {} has changed on the server. downloading newest version...".format(
                        os.path.basename(path)))
                    os.remove(path) # remove outdated file
                    download_file = True
            else:
                logger.warning("no hash sum of {} available, skipping up-to-date check".format(os.path.basename(path)))

    if download_file:
        # does not exist yet -> download file
        from NuRadioReco.utilities.dataservers import download_from_dataserver

        antenna_pattern_name = os.path.splitext(os.path.basename(path))[0]
        remote_path = 'AntennaModels/{name}/{name}.pkl'.format(name=antenna_pattern_name)

        download_from_dataserver(remote_path, path)

    # # does not exist yet -> precalculating WIPLD simulations from raw WIPLD output
    # preprocess_WIPLD(path)
    res = io_utilities.read_pickle(path, encoding='bytes')

    if return_verified:
        return res, verified
    else:
        return res


def parse_AERA_XML_file(path):
    import xml.etree.ElementTree as ET

    if not os.path.exists(path):
        logger.error("AERA antenna file {} not found".format(path))
        raise OSError

    antenna_file = open(path, "r")

    antenna_data = "<antenna>" + antenna_file.read() + "</antenna>"  # add pseudo root element

    # get root element
    root = ET.fromstring(antenna_data)

    # get frequencies and angles
    frequencies_node = root.find("./frequency")
    frequencies = np.array(frequencies_node.text.strip().split(), dtype=float) * units.MHz

    theta_node = root.find("./theta")
    thetas = np.array(theta_node.text.strip().split(), dtype=float) * units.deg

    phi_node = root.find("./phi")
    phis = np.array(phi_node.text.strip().split(), dtype=float) * units.deg

    n_freqs = len(frequencies)
    n_angles = len(phis)

    # get amplitude and phase
    theta_amps = np.zeros((n_freqs, n_angles))
    theta_phases = np.zeros((n_freqs, n_angles))
    phi_amps = np.zeros((n_freqs, n_angles))
    phi_phases = np.zeros((n_freqs, n_angles))

    for iFreq, freq in enumerate(frequencies / units.MHz):
        freq_string = "%.2f" % freq

        theta_amp_node = root.find("./EAHTheta_amp[@idfreq='%s']" % freq_string)

        # check string
        if theta_amp_node is None:
            freq_string = "%.1f" % freq

        theta_amp_node = root.find("./EAHTheta_amp[@idfreq='%s']" % freq_string)
        theta_amps[iFreq] = np.array(theta_amp_node.text.strip().split(), dtype=float) * units.m

        theta_phase_node = root.find("./EAHTheta_phase[@idfreq='%s']" % freq_string)
        theta_phases[iFreq] = np.deg2rad(np.array(theta_phase_node.text.strip().split(" "), dtype=float))

        phi_amp_node = root.find("./EAHPhi_amp[@idfreq='%s']" % freq_string)
        phi_amps[iFreq] = np.array(phi_amp_node.text.strip().split(), dtype=float) * units.m

        phi_phase_node = root.find("./EAHPhi_phase[@idfreq='%s']" % freq_string)
        phi_phases[iFreq] = np.deg2rad(np.array(phi_phase_node.text.strip().split(), dtype=float))

    return frequencies, phis, thetas, phi_amps, phi_phases, theta_amps, theta_phases


def preprocess_AERA(path):
    frequencies, phis, thetas, phi_amps, phi_phases, theta_amps, theta_phases = parse_AERA_XML_file(path)

    n_freqs = len(frequencies)
    n_angles = len(phis)

    def P2R(magnitude, phase):
        return magnitude * np.exp(1j * phase)

    VEL_thetas = P2R(theta_amps, theta_phases)
    VEL_phis = P2R(phi_amps, phi_phases)

    # (angle) -> (freq * angle)
    thetas = np.tile(thetas, n_freqs)
    phis = np.tile(phis, n_freqs)

    # (freq) -> (freq * angles)
    ff = np.repeat(frequencies, n_angles)

    # sort with increasing frequency, increasing phi, and increasing theta
    index = np.lexsort((thetas, phis, ff))
    VEL_thetas = VEL_thetas.flatten()[index]
    VEL_phis = VEL_phis.flatten()[index]

    # (angle) -> (freq * angle)
    theta = np.tile(thetas, n_freqs)[index]
    phi = np.tile(phis, n_freqs)[index]

    # to avoid issues when deviding throw H (H=0 is ignored)
    # |H| < 0.1 should not happen between 30 - 80 MHz
    H_phi = np.where(np.abs(VEL_phis) > 0.01, VEL_phis, 0)
    H_theta = np.where(np.abs(VEL_thetas) > 0.01, VEL_thetas, 0)

    # values for a upwards pointing LPDA with the arm aligned to the magnetic field
    orientation_theta, orientation_phi, rotation_theta, rotation_phi = 0 * units.deg, 0 * units.deg, 90 * units.deg, 90 * units.deg

    fname = os.path.split(os.path.basename(path))[1].replace('.xml', '')
    output_filename = '{}.pkl'.format(os.path.join(path_to_antennamodels, fname, fname))

    directory = os.path.dirname(output_filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(output_filename, 'wb') as fout:
        logger.info('saving output to {}'.format(output_filename))
        pickle.dump([orientation_theta, orientation_phi, rotation_theta, rotation_phi, ff, theta, phi, H_phi, H_theta],
                    fout, protocol=4)


def parse_ARA_file(ara):
    """
    Helper function that parses the ARAsim ASCII files containig antenna responses

    Parameters
    ----------
    ara: string
        path to the file

    Returns
    -------
    ff: array of floats
        frequencies
    thetas: array of floats
        zenith angle of inicdent electric field
    phis: array of floats
        azimuth angle of inicdent electric field
    gains: array of floats
        corresponding linear gain values
    phases: array of floats
        corresponding phases
     """
    with open(ara, 'r') as fin:
        ff = []
        phis = []
        thetas = []
        gains = []
        phases = []
        f = None
        tmp_phi0_lines = []
        for line in fin.readlines():
            if line.strip().startswith('freq'):
                # add phi = 360deg = 0deg to data structure (to allow for a automated interpolation
                f = float(line.replace(" ", "").replace("freq", "").replace(":", "").replace("MHz", ""))
                continue
            if line.strip().startswith('SWR'):
                continue
            if line.strip().startswith('Theta'):
                continue
            ff.append(f * units.MHz)
            theta, phi, gaindB, gain, phase = line.split()
            if float(phi) == 0:
                tmp_phi0_lines.append(line)
            phis.append(float(phi) * units.deg)
            thetas.append(float(theta) * units.deg)
            gains.append(float(gain))
            phases.append(float(phase) * units.deg)
            if float(phi) == 355 and float(theta) == 180:
                for i, tline in enumerate(tmp_phi0_lines):
                    ff.append(f * units.MHz)
                    theta, phi, gaindB, gain, phase = tline.split()
                    if i == 0:
                        logger.debug("{} {} {} {} {} {}".format(f, theta, phi, gaindB, gain, phase))
                    phis.append(360. * units.deg)
                    thetas.append(float(theta) * units.deg)
                    gains.append(float(gain))
                    phases.append(float(phase) * units.deg)
                tmp_phi0_lines = []

        return np.array(ff), np.array(phis), np.array(thetas), np.array(gains), np.array(phases)


def preprocess_ARA(path):
    """
    preprocess an antenna pattern in the ARASim ASCII file format.

    The vector effective length is calculated and
    the output is saved to the NuRadioReco pickle format.

    Parameters
    ----------
    path: string
        the path to the file

    """
    c = constants.c * units.m / units.s
    Z_0 = 119.9169 * np.pi
    split = os.path.split(os.path.dirname(path))
    name = split[1]
    path = split[0]
    orientation_theta, orientation_phi, rotation_theta, rotation_phi, ff, phi, theta, gain, phase = parse_ARA_file(
        os.path.join(path, name, '{}.txt'.format(name)),
        os.path.join(path, name, '{}.orientation'.format(name)))

    wavelength = c / ff
    H_theta = wavelength * (50 / (np.pi * Z_0)) ** 0.5 * gain ** 0.5  # * np.exp(1j * phase)  ## do not use phases, this will screw up the interpolation
    H_phi = H_theta * 1e-3
    output_filename = '{}.pkl'.format(os.path.join(path, name, name))
    with open(output_filename, 'wb') as fout:
        logger.info('saving output to {}'.format(output_filename))
        pickle.dump([orientation_theta, orientation_phi, rotation_theta, rotation_phi, ff, theta, phi, H_phi, H_theta],
                    fout, protocol=4)


def parse_HFSS_file(hfss):
    """
    Helper function that parses the HFSS files containig antenna responses

    Parameters
    ----------
    hfss: string
        path to the file

    Returns
    -------
    ff: array of floats
        frequencies
    thetas: array of floats
        zenith angle of inicdent electric field
    phis: array of floats
        azimuth angle of inicdent electric field
    magnitudes_theta: array of floats
        corresponding logarithmic magnitude values theta component
    magnitudes_phi: array of floats
        corresponding logarithmic magnitude values phi component
    phases_phi: array of floats
        corresponding phases phi component
    phases_theta: array of floats
        corresponding phases theta component
     """
    ff, phi, theta, mag_phi, mag_theta, phase_phi, phase_theta = [], [], [], [], [], [], []
    import re

    with open(hfss, 'r') as csv_file:

        for j, row in enumerate(csv_file.readlines()):

            if j == 0:
                array_names = row.split(',')
            else:
                array = row.split(',')
                for i in range(len(array_names)):
                    if 'Freq' in array_names[i]:
                        freq = array[i]
                    if 'log10(mag(rEPhi))' in array_names[i]:
                        mag_phi.append(float(array[i]))
                        ff.append(float(freq) * units.MHz)

                        p = re.search("Phi='(.+?)deg'", array_names[i])
                        t = re.search("Theta='(.+?)deg'", array_names[i])
                        phi.append(np.deg2rad(int(p.group(1))))
                        theta.append(np.deg2rad(int(t.group(1))))
                    if 'log10(mag(rETheta))' in array_names[i]:
                        mag_theta.append(float(array[i]))
                    if 'ang_rad(rEPhi)' in array_names[i]:
                        phase_phi.append(float(array[i]))
                    if 'ang_rad(rETheta)' in array_names[i]:
                        phase_theta.append(float(array[i]))

        for i in range(len(np.unique(ff)) + 1):
            for arr in [theta, mag_theta, mag_phi, phase_theta, phase_phi, ff, phi]:
                arr[(i - 1) * len(ff) / len(np.unique(ff)):i * len(ff) / len(np.unique(ff))] = [x for _, x in sorted(
                    zip(phi[(i - 1) * len(ff) / len(np.unique(ff)):i * len(ff) / len(np.unique(ff))],
                        arr[(i - 1) * len(ff) / len(np.unique(ff)):i * len(ff) / len(np.unique(ff))]),
                    key=lambda pair: pair[0])]

        return np.array(ff), np.array(phi), np.array(theta), np.array(mag_phi), np.array(mag_theta), np.array(
            phase_phi), np.array(phase_theta)


def preprocess_HFSS(path):
    """
    preprocess an antenna pattern in the HFSS file format. The realized vector effective length is calculated and the output is saved in the NuRadioReco pickle format.

    The vector effective length calculation still needs to be verified.

    The frequencies, theta, phi, magnitude theta, magnitude phi, phase theta and phase phi are read from the csv file and than ordered according to the NuRadioReco format.


    Parameters
    ----------
    path: string
        the path to the file

    """

    split = os.path.split(os.path.dirname(path))
    name = split[1]
    path = split[0]

    ff, phi, theta, mag_phi, mag_theta, phase_phi, phase_theta = parse_HFSS_file(
        (os.path.join(path, name, '{}.csv'.format(name))))
    mag_theta = 10 ** (mag_theta / 10)
    mag_phi = 10 ** (mag_phi / 10)
    gain_theta = 4.0 * np.pi * (mag_theta ** 2) / (2 * 120 * np.pi)
    gain_phi = 4.0 * np.pi * (mag_phi ** 2) / (2 * 120 * np.pi)
    c = constants.c * units.m / units.s
    Z_0 = 119.9169 * np.pi
    wavelength = c / np.array(ff)
    n_index = 1.78

    H_theta = wavelength / n_index ** 0.5 * (50 / (4 * np.pi * Z_0)) ** 0.5 * gain_theta ** 0.5 * np.exp(
        1j * phase_theta)
    H_phi = wavelength / n_index ** 0.5 * (50 / (4 * np.pi * Z_0)) ** 0.5 * gain_phi ** 0.5 * np.exp(1j * phase_phi)

    orientation_theta = 0
    orientation_phi = 0
    rotation_theta = 0
    rotation_phi = 0

    output_filename = '{}.pkl'.format(os.path.join(path, name, name))

    with open(output_filename, 'wb') as fout:
        logger.info('saving output to {}'.format(output_filename))
        pickle.dump([orientation_theta, orientation_phi, rotation_theta, rotation_phi, ff, theta, phi, H_phi, H_theta],
                    fout, protocol=4)


def preprocess_XFDTD(path):
    """
    preprocess an antenna pattern in the XFDTD file format. The realized vector effective length is calculated and
    the output is saved to the NuRadioReco pickle format.

    Parameters
    ----------
    path: string
        the path to the file

    """
    split = os.path.split(os.path.dirname(path))
    name = split[1]
    path = split[0]

    import yaml
    with open(os.path.join(path, name, '{}.yaml'.format(name))) as fin:
        info = yaml.load(fin)
        orientation_theta, orientation_phi = hp.cartesian_to_spherical(*info['boresight_direction'])
        rotation_theta, rotation_phi = hp.cartesian_to_spherical(*info['orientation'])
        n_index = info['n']

        c = constants.c * units.m / units.s
        Z_0 = 119.9169 * np.pi
        ff, phi, theta, gain, phase = parse_ARA_file(os.path.join(path, name, '{}.txt'.format(name)))
        wavelength = c / ff
        H = wavelength / n_index ** 0.5 * (50 / (4 * np.pi * Z_0)) ** 0.5 * gain ** 0.5 * np.exp(1j * phase)
        if info['type'] == 'Vpol':
            H_theta = H
            H_phi = H * 1e-6
        elif info['type'] == 'Hpol':
            H_theta = H * 1e-6
            H_phi = H
        else:
            logger.error("antenna type {} not understood".format(info['type']))
            raise NotImplementedError("antenna type {} not understood".format(info['type']))

        output_filename = '{}.pkl'.format(os.path.join(path, name, name))
        with open(output_filename, 'wb') as fout:
            logger.info('saving output to {}'.format(output_filename))
            pickle.dump(
                [orientation_theta, orientation_phi, rotation_theta, rotation_phi, ff, theta, phi, H_phi, H_theta],
                fout, protocol=4)


def parse_LOFAR_txt_file(path_theta, path_phi):
    """
    Extract the values from a simulation file for the LOFAR LBA antenna model.

    Parameters
    ----------
    path_theta : str
        Path to the file containing the values for the theta component
    path_phi : str
        Path to the file containing the values for the phi component

    """
    freq, theta, phi, real_theta, imaginary_theta = np.genfromtxt(path_theta, skip_header=1).T
    freq2, theta2, phi2, real_phi, imaginary_phi = np.genfromtxt(path_phi, skip_header=1).T

    if not np.all(freq == freq2) or not np.all(theta == theta2) or not np.all(phi == phi2):
        raise ValueError("Values in theta and phi files do not match")

    # Convert units to NRR system
    freq *= units.MHz
    theta *= units.deg
    phi *= units.deg

    # Add the weird -1 to the theta component
    real_theta *= -1
    imaginary_theta *= -1

    return freq, theta, phi, real_theta, imaginary_theta, real_phi, imaginary_phi


def preprocess_LOFAR_txt(directory, ant='LBA', orientation=None):
    """
    Function to process the TXT files from the old LOFAR antenna model (only tested for LBA). The paths to these
    files is currently hardcoded. Because of a weird issue which requires minus signs to be added for the X and Y
    dipoles separately, the orientation can be specified to create separate antenna models for each. If the
    orientation is not set, the values for the Y dipole are returned.

    Parameters
    ----------
    directory : str
        Path to where the text files are stored
    ant : str, default='LBA'
        The antenna type, either LBA or HBA (not tested)
    orientation : str, default=None
        If set, must be either X or Y.
    """
    path_theta = os.path.join(directory, f'{ant}_Vout_theta.txt')
    path_phi = os.path.join(directory, f'{ant}_Vout_phi.txt')

    frequencies, thetas, phis, theta_real, theta_imag, phi_real, phi_imag = parse_LOFAR_txt_file(path_theta, path_phi)

    if orientation == 'X':
        for ar in [theta_real, theta_imag, phi_real, phi_imag]:
            ar *= -1

    VEL_thetas = theta_real + 1j * theta_imag
    VEL_phis = phi_real + 1j * phi_imag

    # sort with increasing frequency, increasing phi, and increasing theta
    index = np.lexsort((thetas, phis, frequencies))
    VEL_thetas = VEL_thetas.flatten()[index]
    VEL_phis = VEL_phis.flatten()[index]

    # (angle) -> (freq * angle)
    theta = thetas[index]
    phi = phis[index]

    # TODO: is the correct calculation of VEL? Felix wrote that for AERA, |H| < 0.1 should not happen...
    H_phi = VEL_phis
    H_theta = VEL_thetas

    # values for an upright LBA antenna aligned along E-W
    orientation_theta, orientation_phi, rotation_theta, rotation_phi = \
        90 * units.deg, 0 * units.deg, 0 * units.deg, 0 * units.deg

    if orientation is not None:
        fname = f'LOFAR_{ant}_{orientation}'
    else:
        fname = f'LOFAR_{ant}'
    output_filename = '{}.pkl'.format(os.path.join(path_to_antennamodels, fname, fname))

    directory = os.path.dirname(output_filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(output_filename, 'wb') as fout:
        logger.info('saving output to {}'.format(output_filename))
        # Notice the ordering!
        pickle.dump([orientation_theta, orientation_phi, rotation_theta, rotation_phi,
                     frequencies, theta, phi, H_phi, H_theta],
                    fout, protocol=4)

def preprocess_FEKO_mat(path, polarization='X', downscale_freq=1, downscale_zenith=4, downscale_azimuth=4):
    """
    used to convert FEKO_AAVS2_single_elem_50ohm_50_350MHz_{polarization}pol.mat for the SKALA4 antenna to a pickle file

    The file contains the embedded element simulation of the SKALA4 antenna in the frequency range of 50-350 MHz.
    The values correspond to the far-field emission of this antenna; it is converted
    to the realized vector effective length for a receiving antenna using Eq. 6 in [1]_.

    Parameters
    ----------
    directory : str
        The path to the directory where the files are stored

    polarization : str, default='X'
        X polarization is the antenna in east-west orientation, Y polarization is the antenna in north-south orientation.

    downscale_freq : int, default: 1
        The downscaling factor for the frequency spacing.
        The native frequency spacing is 1 MHz, and the default
        downscaling factor is 1 (no downscaling).

    downscale_zenith : int, default: 4
        The downscaling factor for the zenith spacing.
        The native zenith spacing is 0.5 degrees, and the default
        downscaling factor is 4, resulting in a spacing of 2 degrees.

    downscale_azimuth : int, default: 4
        The downscaling factor for the azimuth spacing.
        The native azimuth spacing is 0.5 degrees, and the default
        downscaling factor is 4, resulting in a spacing of 2 degrees.


    References
    ----------
    .. [1] https://arxiv.org/abs/2412.01699
    """

    input_file = os.path.join(path, f'FEKO_AAVS2_single_elem_50ohm_50_350MHz_{polarization}pol.mat')
    data = scipy.io.loadmat(input_file)
    # the data format is 721 x 181 x 301 (Phi, theta, freq)
    # NuRadio (for antenna models) expects the order (freq, phi, theta), so we have to move some axes
    Ephi = data['Ephi'].transpose(2, 0, 1)
    Etheta = data['Etheta'].transpose(2, 0, 1)

    # the data is stored in 1 MHz and 0.5 degree spacing
    freqs_unique = np.linspace(50, 350, 301) * units.MHz
    phis_unique = np.linspace(0, 360, 721) * units.deg
    thetas_unique = np.linspace(0, 90, 181) * units.deg

    freq, phi, theta = np.meshgrid(
        freqs_unique, phis_unique, thetas_unique, indexing='ij')

    # downscale from native spacing if required
    if not np.all(np.array([downscale_freq, downscale_zenith, downscale_azimuth]) == 1):
        mask = np.zeros_like(phi).astype(int)
        mask[np.arange(0, len(freqs_unique), downscale_freq), :, :] += 1
        mask[:, np.arange(0, len(phis_unique), downscale_azimuth), :] += 1
        mask[:, :, np.arange(0, len(thetas_unique), downscale_zenith)] += 1
        mask = mask > 2 # equivalent to applying the three masks successively

        Ephi = Ephi[mask]
        Etheta = Etheta[mask]
        phi = phi[mask]
        theta = theta[mask]
        freq = freq[mask]

        logger.status(f'Rescaling SKALA4 antenna from shape ({mask.shape}) to {Ephi.shape}...')

    lambda_0 = (constants.speed_of_light * units.m / units.s) / freq # wavelength
    eta_0 = np.sqrt(constants.mu_0 / constants.epsilon_0) * units.ohm # free space impedance
    Z_L = 50 * units.ohm # we assume a 50 Ohm amplifier
    vel_theta = -2.j * lambda_0 * Z_L / eta_0 * Etheta
    del Etheta # free up some memory?
    vel_phi = -2.j * lambda_0 * Z_L / eta_0  * Ephi
    del Ephi # free up some memory?

    orientation_theta = 0
    orientation_phi = 0
    rotation_theta = 90 * units.deg

    if polarization == 'X':
        # use this angles and name SKALA_v4_Xpol to have your channel in east-west orientation
        rotation_phi = 90 * units.deg
    if polarization == 'Y':
        # use this angles and name SKALA_v4_Ypol to have your channel in north-south orientation
        rotation_phi = 180 * units.deg

    fname = f'SKALA_v4_{polarization}pol'
    output_filename = "{}.pkl".format(os.path.join(path_to_antennamodels, fname, fname))

    directory = os.path.dirname(output_filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(output_filename, 'wb') as fout:
        logger.warning('saving antenna output to {}'.format(output_filename))
        pickle.dump([orientation_theta, orientation_phi, rotation_theta, rotation_phi,
                     freq, theta, phi, vel_phi, vel_theta],
                    fout, protocol=4)

def get_onsky_rotation(zenith, azimuth):
    """
    Rotation matrix from the ground (x, y, z) into the on-sky (eR, eTheta, ePhi) system

    Equivalent to ``radiotools.coordinatesystems.cstrafo`` but constructs only the
    transformation needed here. The matrix is orthonormal, i.e. its transpose describes
    the inverse transformation.

    Parameters
    ----------
    zenith, azimuth: float
        direction defining the on-sky coordinate system

    Returns
    -------
    rotation: array of floats
        the 3x3 rotation matrix
    """
    ct, st = np.cos(zenith), np.sin(zenith)
    cp, sp = np.cos(azimuth), np.sin(azimuth)
    return np.array([
        [st * cp, st * sp, ct],
        [ct * cp, ct * sp, -st],
        [-sp, cp, 0.]])


def get_orthonormal_basis(theta1, phi1, theta2, phi2, description):
    """
    Basis spanned by two (almost) perpendicular directions and their cross product

    Parameters
    ----------
    theta1, phi1: float
        zenith and azimuth angle of the first direction (for antennas: the boresight)
    theta2, phi2: float
        zenith and azimuth angle of the second direction (for antennas: perpendicular
        to the boresight, e.g. the normal of the plane containing the tines of an LPDA)
    description: string
        what is being defined, used for the error message

    Returns
    -------
    basis: array of floats
        the 3x3 matrix of the three basis vectors
    """
    e1 = hp.spherical_to_cartesian(theta1, phi1)
    e2 = hp.spherical_to_cartesian(theta2, phi2)
    e3 = np.cross(e1, e2)

    if np.linalg.norm(e3) < 0.9:  # the two directions are not perpendicular enough
        logger.error("orientation of antenna not properly defined in {}".format(description))
        raise AssertionError("orientation of antenna not properly defined in {}".format(description))

    # the two directions are only required to be almost perpendicular and the orientation
    # stored in some antenna models is off by a few 0.01 deg. Without orthonormalising, the
    # transformation between the two bases is not a rotation and rescales the VEL slightly.
    e2 = e2 - np.dot(e2, e1) * e1
    e2 /= np.linalg.norm(e2)

    return np.array([e1, e2, np.cross(e1, e2)])


class AntennaPatternBase:
    """
    base class of utility class that handles access and buffering to antenna pattern
    """

    def _get_antenna_rotation(self, orientation_theta, orientation_phi, rotation_theta, rotation_phi):
        """
        Rotation from the coordinate system of the antenna simulation into the one of
        the antenna as deployed in the field, and its inverse

        The result only depends on the requested orientation, of which a detector has very
        few, so it is buffered.

        Parameters
        ----------
        orientation_theta, orientation_phi: float
            orientation (boresight) of the antenna in the field, see
            `AntennaPatternBase.get_antenna_response_vectorized`
        rotation_theta, rotation_phi: float
            rotation of the antenna in the field, see
            `AntennaPatternBase.get_antenna_response_vectorized`

        Returns
        -------
        rotation, inverse_rotation: array of floats
            the 3x3 rotation matrix and its inverse
        polar_roll: bool
            True if the rotation is a pure roll about the polar axis, see
            `AntennaPatternBase.get_antenna_response_vectorized`
        """
        orientation = (orientation_theta, orientation_phi, rotation_theta, rotation_phi)
        if orientation not in self._antenna_rotations:
            simulated = get_orthonormal_basis(
                self._orientation_theta, self._orientation_phi,
                self._rotation_theta, self._rotation_phi, "the antenna model")
            deployed = get_orthonormal_basis(*orientation, "the detector description")

            # both bases are orthonormal, hence the inverse is the transpose
            rotation = np.matmul(simulated.T, deployed)
            polar_roll = (np.allclose(rotation[2], [0., 0., 1.])
                          and np.allclose(rotation[:, 2], [0., 0., 1.]))
            self._antenna_rotations[orientation] = (rotation, rotation.T, polar_roll)

        return self._antenna_rotations[orientation]

    def _get_theta_and_phi(self, zenith, azimuth, orientation_theta, orientation_phi,
                           rotation_theta, rotation_phi):
        """
        Transform an incoming signal direction from the NuRadio into the antenna
        simulation coordinate system, taking the orientation of the antenna as deployed
        in the field into account

        Parameters
        ----------
        zenith, azimuth: float
            incoming signal direction in the NuRadio coordinate system
        orientation_theta, orientation_phi, rotation_theta, rotation_phi: float
            orientation of the antenna in the field, see
            `AntennaPatternBase.get_antenna_response_vectorized`

        Returns
        -------
        theta, phi: float
            the same direction in the coordinate system of the antenna simulation
        """
        rotation, _, _ = self._get_antenna_rotation(
            orientation_theta, orientation_phi, rotation_theta, rotation_phi)

        incoming_direction = hp.spherical_to_cartesian(zenith, azimuth)
        theta, phi = hp.cartesian_to_spherical(*np.dot(rotation, incoming_direction.T).T)

        logger.debug("zen/az {:.0f} {:.0f} transform to {:.0f} {:.0f}".format(
            zenith / units.deg, azimuth / units.deg, theta / units.deg, phi / units.deg))

        return theta, phi

    def get_antenna_response_vectorized(self, freq, zenith, azimuth, orientation_theta, orientation_phi, rotation_theta,
                                        rotation_phi):
        """
        get the antenna response for a specific frequency, zenith and azimuth angle

        All angles are specified in the NuRadio coordinate system. All units are in NuRadio default units

        Parameters
        ----------
        freq : float or array of floats
            frequency
        zenith : float
            zenith angle of incoming signal direction
        azimuth : float
            azimuth angle of incoming signal direction
        orientation_theta: float
            orientation of the antenna, as a zenith angle (0deg is the zenith, 180deg is straight down); for LPDA: outward along boresight; for dipoles: upward along axis of azimuthal symmetry
        orientation_phi: float
            orientation of the antenna, as an azimuth angle (counting from East counterclockwise); for LPDA: outward along boresight; for dipoles: upward along axis of azimuthal symmetry
        rotation_theta: float
            rotation of the antenna, is perpendicular to 'orientation', for LPDAs: vector perpendicular to the plane containing the the tines
        rotation_phi: float
            rotation of the antenna, is perpendicular to 'orientation', for LPDAs: vector perpendicular to the plane containing the the tines

        Returns
        -------
        VEL: dictonary of complex arrays
            theta and phi component of the vector effective length, both components
            are complex floats or arrays of complex floats
            of the same length as the frequency input
        """
        freq = np.atleast_1d(freq)
        if self._notfound:
            return {'theta': np.ones(len(freq), dtype=complex),
                    'phi': np.ones(len(freq), dtype=complex)}

        theta, phi = self._get_theta_and_phi(
            zenith, azimuth, orientation_theta, orientation_phi, rotation_theta, rotation_phi)

        VEL_theta_raw, VEL_phi_raw = self._get_antenna_response_vectorized_raw(freq, theta, phi)

        _, inverse_antenna_rotation, polar_roll = self._get_antenna_rotation(
            orientation_theta, orientation_phi, rotation_theta, rotation_phi)

        # A roll of the antenna about the polar axis only shifts the azimuth, and the on-sky
        # unit vectors at the shifted azimuth are rolled by exactly the same angle. Both
        # cancel for every direction, i.e. the two on-sky systems already coincide.
        if polar_roll:
            return {'theta': VEL_theta_raw, 'phi': VEL_phi_raw}

        # The eTheta and ePhi unit vectors of the antenna simulation and of NuRadio point in
        # different directions, so rotate the VEL from the one on-sky system into the other
        # by going through the (cartesian) ground coordinate system.
        rotation = (get_onsky_rotation(zenith, azimuth)
                    @ inverse_antenna_rotation
                    @ get_onsky_rotation(theta, phi).T)

        # the eR component of the VEL is zero and the one of the result is not needed,
        # hence only the eTheta/ePhi block of the rotation contributes
        VEL = np.matmul(rotation[1:, 1:], np.array([VEL_theta_raw, VEL_phi_raw]))

        return {'theta': VEL[0], 'phi': VEL[1]}


class AntennaPattern(AntennaPatternBase):
    """
    Utility class that handles access and buffering to simulated antenna pattern.
    The class accesses the NuRadioReco pickle format file which contains the preprocessed antenna pattern.

    The pickle file contains 9 lists of the following elements:

    orientation_theta: float
        orientation of the antenna, as a zenith angle (0deg is the zenith, 180deg is straight down); for LPDA: outward along boresight; for dipoles: upward along axis of azimuthal symmetry
    orientation_phi: float
        orientation of the antenna, as an azimuth angle (counting from East counterclockwise); for LPDA: outward along boresight; for dipoles: upward along axis of azimuthal symmetry
    rotation_theta: float
        rotation of the antenna, is perpendicular to 'orientation', for LPDAs: vector perpendicular to the plane containing the the tines
    rotation_phi: float
        rotation of the antenna, is perpendicular to 'orientation', for LPDAs: vector perpendicular to the plane containing the the tines
    ff: array of floats
        array of frequencies
    thetas: array of floats
        zenith angle of inicdent electric field
    phis: array of floats
        azimuth angle of incident electric field
    H_phi: array of floats
        the complex realized vector effective length of the ePhi polarization component
    H_theta: array of floats
        the complex realized vector effective length of the eTheta polarization component

    The three angular/spectral axes are sampled on a regular (but not necessarily equally
    spaced) grid, i.e. the VEL is stored as a ``(n_freqs, n_phi, n_theta)`` cube.
    """

    def __init__(self, antenna_model, path=path_to_antennamodels,
                 interpolation_method='complex', do_consistency_check=True):
        """

        Parameters
        ----------
        antenna_model: string
            name of antenna model
        path: string
            path to folder containing the antenna models
        interpolation_mode: string
            specify in which domain the interpolation should be performed, can be either

            * 'complex' (default) interpolate real and imaginary part of vector effective length
            * 'magphase' interpolate magnitude and phase of vector effective length

        consistency_check: bool (default: True)
            If True, the consistency of the antenna response is checked but only if antenna
            file could not be verified (via hash sum).
        """

        self._name = antenna_model
        self._interpolation_method = interpolation_method
        self._antenna_rotations = {}

        t0 = time()
        filename = os.path.join(path, antenna_model, "{}.pkl".format(antenna_model))
        self._notfound = False
        try:
            (self._orientation_theta, self._orientation_phi, self._rotation_theta, self._rotation_phi, \
                ff, thetas, phis, H_phi, H_theta), verified = get_pickle_antenna_response(filename, return_verified=True)

        except IOError:
            self._notfound = True
            logger.error("antenna response for {} not found".format(antenna_model))
            raise FileNotFoundError("antenna response for {} not found".format(antenna_model))

        self.frequencies = np.unique(ff)
        self.theta_angles = np.unique(thetas)
        self.phi_angles = np.unique(phis)

        self.n_freqs = len(self.frequencies)
        self.n_theta = len(self.theta_angles)
        self.n_phi = len(self.phi_angles)

        self.frequency_lower_bound, self.frequency_upper_bound = self.frequencies[[0, -1]]
        self.theta_lower_bound, self.theta_upper_bound = self.theta_angles[[0, -1]]
        self.phi_lower_bound, self.phi_upper_bound = self.phi_angles[[0, -1]]
        logger.debug("{} thetas from {} to {}".format(
            self.n_theta, self.theta_lower_bound, self.theta_upper_bound))
        logger.debug("{} phis from {} to {}".format(
            self.n_phi, self.phi_lower_bound, self.phi_upper_bound))

        # allows the faster index calculation in `get_bracketing_indices`
        self._equidistant_freqs = is_equidistant(self.frequencies)
        self._equidistant_theta = is_equidistant(self.theta_angles)
        self._equidistant_phi = is_equidistant(self.phi_angles)

        if do_consistency_check and not verified:
            logger.status("Performing consistency check on antenna response ...")
            self._check_grid_ordering(ff, thetas, phis)

        # both components in one array so that a single gather serves both, VEL_theta and
        # VEL_phi are views into it
        grid_shape = (self.n_freqs, self.n_phi, self.n_theta)
        self.VEL = np.array([H_theta.reshape(grid_shape), H_phi.reshape(grid_shape)])
        self.VEL_theta, self.VEL_phi = self.VEL

        logger.status('Loading antenna file {} took {:.1f} seconds'.format(antenna_model, time() - t0))

    def _check_grid_ordering(self, ff, thetas, phis):
        """
        Check that the flat arrays of the antenna file iterate over theta first, then phi,
        then frequency, which is what reshaping them into the VEL cube assumes
        """
        expected = {
            "frequency": (ff, np.repeat(self.frequencies, self.n_phi * self.n_theta)),
            "phi angle": (phis, np.tile(np.repeat(self.phi_angles, self.n_theta), self.n_freqs)),
            "theta angle": (thetas, np.tile(self.theta_angles, self.n_freqs * self.n_phi)),
        }

        for name, (stored, ordered) in expected.items():
            if not np.array_equal(stored, ordered):
                logger.error("{} is not ordered as expected in {}".format(name, self._name))
                raise Exception("{} is not ordered as expected in {}".format(name, self._name))

    def _get_antenna_response_vectorized_raw(self, freq, theta, phi):
        """
        Get the vector effective length in the coordinate system of the antenna simulation

        Trilinearly interpolates the VEL cube at the requested position. Frequencies
        outside of the simulated band return 0, directions outside of the simulated
        solid angle return 0 for all frequencies.

        Parameters
        ----------
        freq: array of floats
            frequencies at which to evaluate the response
        theta, phi: float
            direction in the coordinate system of the antenna simulation

        Returns
        -------
        VEL_theta, VEL_phi: arrays of complex floats
            the two components of the vector effective length
        """
        freq = np.atleast_1d(freq)
        phi = self.phi_lower_bound + (phi - self.phi_lower_bound) % (2 * np.pi)

        # avoid that rounding pushes a request at exactly 0 or 180 deg out of the grid
        if hp.is_equal(theta, self.theta_upper_bound, rel_precision=1e-5):
            theta = self.theta_upper_bound
        if hp.is_equal(theta, self.theta_lower_bound, rel_precision=1e-5):
            theta = self.theta_lower_bound

        if not (self.phi_lower_bound <= phi <= self.phi_upper_bound
                and self.theta_lower_bound <= theta <= self.theta_upper_bound):
            logger.warning("theta or phi out of range for {}, returning (0, 0j)".format(self._name))
            logger.debug("theta bounds {}, {}, {}".format(self.theta_lower_bound, theta, self.theta_upper_bound))
            logger.debug("phi bounds {}, {}, {}".format(self.phi_lower_bound, phi, self.phi_upper_bound))
            return np.zeros((2, len(freq)), dtype=complex)

        in_band = ((freq >= self.frequency_lower_bound) & (freq <= self.frequency_upper_bound))
        if not np.any(in_band):
            return np.zeros((2, len(freq)), dtype=complex)

        # the grids are not necessarily equally spaced (e.g. the theta grid of
        # RNOG_vpol_4inch_center_n1.73), hence look up the bracketing nodes instead of
        # calculating their indices from the grid boundaries
        i_theta = np.array(get_bracketing_indices(theta, self.theta_angles, self._equidistant_theta))
        i_phi = np.array(get_bracketing_indices(phi, self.phi_angles, self._equidistant_phi))

        # Collapse the two angular axes first, over the frequency nodes that the request
        # spans: theta and phi have a single bracket while every requested frequency has its
        # own, so this interpolates a few hundred grid points instead of 8 per frequency bin.
        first = max(int(np.searchsorted(self.frequencies, freq[in_band].min(), "right")) - 1, 0)
        last = min(int(np.searchsorted(self.frequencies, freq[in_band].max(), "left")) + 1, self.n_freqs)

        method = self._interpolation_method
        VEL = self.VEL[:, first:last][:, :, i_phi[:, None], i_theta[None, :]]

        if method == 'complex':
            # the bilinear interpolation written as a single weighted sum of the four
            # corners, which is ~2x faster than interpolating one axis after the other
            w_phi = get_interpolation_weight(phi, *self.phi_angles[i_phi])
            w_theta = get_interpolation_weight(theta, *self.theta_angles[i_theta])
            VEL = (VEL[..., 0, 0] * ((1 - w_phi) * (1 - w_theta))
                   + VEL[..., 0, 1] * ((1 - w_phi) * w_theta)
                   + VEL[..., 1, 0] * (w_phi * (1 - w_theta))
                   + VEL[..., 1, 1] * (w_phi * w_theta))
        else:
            VEL = interpolate_linear(
                phi, *self.phi_angles[i_phi], VEL[..., 0, :], VEL[..., 1, :], method)
            VEL = interpolate_linear(
                theta, *self.theta_angles[i_theta], VEL[..., 0], VEL[..., 1], method)

        # ... and only then interpolate along the frequency axis, shape (component, n_freq)
        VEL = self._interpolate_frequency(freq, self.frequencies[first:last], VEL)

        VEL[:, ~in_band] = 0
        return VEL[0], VEL[1]

    def _interpolate_frequency(self, freq, nodes, VEL):
        """
        Interpolate the vector effective length at the frequency nodes onto ``freq``

        Parameters
        ----------
        freq: array of floats
            the requested frequencies
        nodes: array of floats
            the frequency nodes VEL is given at
        VEL: array of complex floats
            the vector effective length, shape (component, len(nodes))

        Returns
        -------
        VEL: array of complex floats
            the interpolated vector effective length, shape (component, len(freq))
        """
        if self._interpolation_method == 'complex':
            # np.interp does the bracketing internally, in C and for the whole vector at once
            return np.array([np.interp(freq, nodes, VEL[0]), np.interp(freq, nodes, VEL[1])])

        i_freq = np.array(get_bracketing_indices(freq, nodes, is_equidistant(nodes)))
        return interpolate_linear(
            freq, *nodes[i_freq], VEL[:, i_freq[0]], VEL[:, i_freq[1]], self._interpolation_method)


class AntennaPatternAnalytic(AntennaPatternBase):
    """
    utility class that handles access and buffering to analytic antenna pattern
    """

    # default low frequency cutoff (peak frequency for the HPol) and maximum VEL, chosen
    # such that the model approximates the corresponding simulated antenna model
    _defaults = {
        'analytic_LPDA': (110 * units.MHz, 0.55 * units.m),   # createLPDA_100MHz_InfFirn_n1.4
        'analytic_VPol': (220 * units.MHz, 0.18 * units.m),   # RNOG_vpol_v3_5inch_center_n1.74
        'analytic_HPol': (500 * units.MHz, 0.055 * units.m),  # RNOG_hpol_v4_8inch_center_n1.74
    }

    def __init__(self, antenna_model, cutoff_freq=None, max_VEL=None):
        """
        Parameters
        ----------
        antenna_model: string
            Name of antenna model. Current implemented models are 'analytic_LPDA',
            'analytic_VPol', and 'analytic_HPol'. By using the default values of cutoff_freq
            and max_VEL, these models approximate the createLPDA_100MHz_InfFirn_n1.4,
            RNOG_vpol_v3_5inch_center_n1.74, and RNOG_hpol_v4_8inch_center_n1.74
            antenna models, respectively.
        cutoff_freq: float
            Sets the low frequency cutoff for the LPDA and VPol models, and the peak
            frequency for the HPol model. Setting cutoff_freq=None, the default values are
            used:
            'analytic_LPDA': 110 MHz,
            'analytic_VPol': 220 MHz,
            'analytic_HPol': 500 MHz.
        max_VEL: float
            Sets the maximum value of the vector effective length for the antenna models.
            Setting max_VEL=None, the default values are used:
            'analytic_LPDA': 0.55 m,
            'analytic_VPol': 0.18 m,
            'analytic_HPol': 0.055 m.
        """
        if antenna_model not in self._defaults:
            logger.error("analytic antenna model {} is not implemented".format(antenna_model))
            raise ValueError("analytic antenna model {} is not implemented".format(antenna_model))

        self._notfound = False
        self._model = antenna_model
        self._antenna_rotations = {}

        default_cutoff_freq, default_max_VEL = self._defaults[antenna_model]
        self._cutoff_freq = default_cutoff_freq if cutoff_freq is None else cutoff_freq
        self._max_VEL = default_max_VEL if max_VEL is None else max_VEL

        # all dummy models point towards z, the LPDA has its tines in the y-z plane
        self._orientation_theta = 0 * units.deg
        self._orientation_phi = 0 * units.deg
        self._rotation_theta = 90 * units.deg
        self._rotation_phi = 0 * units.deg

    def parametric_phase(self, freq, phase_type='theoretical'):
        """
        Phase of the analytic antenna models as a function of frequency

        Parameters
        ----------
        freq: array of floats
            frequencies at which to evaluate the phase
        phase_type: string
            one of 'theoretical', 'frontlobe_lpda', 'side_lpda', 'back_lpda',
            'VPol_third_order' or 'HPol_third_order'

        Returns
        -------
        phase: array of floats
            the phase in radians
        """
        # The third order parametrizations: the 1st, 2nd, and 3rd order parameters are obtained
        # from a 2nd order polynomial fit to the slope of the unwrapped complex phase of the
        # corresponding simulated antenna pattern. The constant term is obtained by fitting the
        # resulting analytic antenna pattern (with the offset set to 0) in the time domain to
        # the simulated antenna pattern in the time domain.
        if phase_type == 'theoretical':
            tau = 0.75             # ratio of two elements
            f = 1000. * units.MHz  # maximum frequency
            return np.pi / np.log(tau) * np.log(freq / f) - 60
        elif phase_type == 'frontlobe_lpda':
            phase = 100 * (freq - 400 * units.MHz) ** 2 - 20
            above = freq > 400 * units.MHz
            phase[above] -= 0.00007 * (freq[above] - 400 * units.MHz) ** 2
            return phase
        elif phase_type == 'side_lpda':
            return 40 * (freq - 950 * units.MHz) ** 2 - 40
        elif phase_type == 'back_lpda':
            return 50 * (freq - 950 * units.MHz) ** 2 - 50
        elif phase_type == 'VPol_third_order':
            return 2.086 - 117.917 * freq + 74.567 / 2 * freq ** 2 - 64.343 / 3 * freq ** 3
        elif phase_type == 'HPol_third_order':
            return 0.321 - 11.400 * freq + 39.590 / 2 * freq ** 2 - 38.181 / 3 * freq ** 3

        logger.error("phase type {} not implemented".format(phase_type))
        raise NotImplementedError("phase type {} not implemented".format(phase_type))

    def _gain_to_vel(self, freq, gain, low_frequency_cutoff=True):
        """
        Convert a gain into a vector effective length, apply the low frequency cutoff
        and normalize to the maximum VEL of the model

        Parameters
        ----------
        freq: array of floats
            frequencies at which to evaluate the response
        gain: array of floats
            the gain at those frequencies
        low_frequency_cutoff: bool
            if True, suppress the response below ``cutoff_freq`` with a half Hann window

        Returns
        -------
        VEL: array of floats
            the vector effective length
        """
        in_band = freq > 0
        VEL = np.zeros_like(freq)
        VEL[in_band] = np.sqrt(gain[in_band]) / freq[in_band]

        if low_frequency_cutoff:
            i_cutoff = np.argmax(freq > self._cutoff_freq)
            VEL[:i_cutoff] *= hann(2 * i_cutoff)[:i_cutoff]

        VEL[in_band] *= self._max_VEL / np.max(VEL[in_band])
        return VEL

    def _get_antenna_response_vectorized_raw(self, freq, theta, phi, group_delay=True):
        """
        Get the vector effective length in the coordinate system of the antenna model

        Parameters
        ----------
        freq: array of floats
            frequencies at which to evaluate the response
        theta, phi: float
            direction in the coordinate system of the antenna model
        group_delay: bool
            if True, apply the parametrized phase of the model

        Returns
        -------
        VEL_theta, VEL_phi: arrays of floats or complex floats
            the two components of the vector effective length
        """
        in_band = freq > 0

        if self._model == 'analytic_LPDA':
            # Flat gain as function of frequency
            VEL = self._gain_to_vel(freq, np.ones_like(freq))
            VEL_theta = VEL * np.cos(theta) * np.sin(phi) * np.cos(theta / 2)
            VEL_phi = VEL * np.cos(theta / 2) * np.cos(phi)

            if theta <= 45 * units.deg:
                phase_type = "frontlobe_lpda"
            elif theta <= 90 * units.deg:
                phase_type = "side_lpda"
            else:
                phase_type = "back_lpda"

        elif self._model == 'analytic_VPol':
            gain = np.ones_like(freq)
            gain[in_band] /= np.sqrt(freq[in_band])  # frequency dependent gain fall-off
            VEL_theta = self._gain_to_vel(freq, gain) * np.sin(theta)
            VEL_phi = np.zeros_like(freq)
            phase_type = "VPol_third_order"

        elif self._model == 'analytic_HPol':
            # cos^2 frequency dependency (peaking at cutoff_freq) and sin^2(theta) directivity
            VEL_phi = np.zeros_like(freq)
            VEL_phi[in_band] = np.sin(freq[in_band] / self._cutoff_freq * np.pi / 2) ** 2
            VEL_phi[freq > 2 * self._cutoff_freq] = 0
            VEL_phi[in_band] *= self._max_VEL / np.max(VEL_phi[in_band])
            VEL_phi *= np.sin(theta) ** 2
            VEL_theta = np.zeros_like(freq)
            phase_type = "HPol_third_order"

        if group_delay:
            phase = np.exp(1j * self.parametric_phase(freq, phase_type))
            VEL_theta = VEL_theta * phase
            VEL_phi = VEL_phi * phase

        return VEL_theta, VEL_phi


class AntennaPatternProvider(object):
    """
    Provider class for antenna pattern. The usage of antenna pattern through this class ensures
    that an antenna pattern is loaded only once into memory which takes a significant time and
    occupies a significant amount of memory.
    """
    __instance = None

    def __new__(cls, *args, **kwargs):
        if AntennaPatternProvider.__instance is None:
            AntennaPatternProvider.__instance = object.__new__(cls)
        return AntennaPatternProvider.__instance

    def __init__(self, log_level=logging.NOTSET):
        """
        Parameters
        ----------
        log_level: int
            the log level of the antenna pattern logger
        """
        logger.setLevel(log_level)

        # the class is a singleton, only set up the buffer on the very first instantiation
        # (otherwise every `AntennaPatternProvider()` would discard the loaded patterns)
        if hasattr(self, "_open_antenna_patterns"):
            return

        self._open_antenna_patterns = {}
        self._antenna_model_replacements = {}

        antenna_directory = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(antenna_directory, 'antenna_model_replacements.json')
        if os.path.exists(filename):
            with open(filename, 'r') as fin:
                self._antenna_model_replacements = json.load(fin)

    def load_antenna_pattern(self, name, **kwargs):
        """
        loads an antenna pattern and returns the antenna pattern class

        Parameters
        ----------
        name: string
            the name of the antenna pattern
        **kwargs: dict
            key word arguments that are passed to the init function of the `AntennaPattern` class (see
            documentation of this class for further information)
        """
        if name in self._antenna_model_replacements:
            replacement = self._antenna_model_replacements[name]
            if not self._is_buffered(replacement):
                logger.status("local replacement of antenna model requsted: replacing {} with {}".format(
                    name, replacement))

            name = replacement

        # the keyword arguments are part of the identity of a pattern: a model requested
        # with different arguments is a different antenna and must not be served from the buffer
        key = (name,) + tuple(sorted((k, repr(v)) for k, v in kwargs.items()))

        if key not in self._open_antenna_patterns:
            if self._is_buffered(name):
                logger.warning(
                    "antenna model {} is already buffered with different arguments, "
                    "loading a second copy for {}".format(name, kwargs))

            if name.startswith("analytic"):
                self._open_antenna_patterns[key] = AntennaPatternAnalytic(name, **kwargs)
                logger.info("loading analytic antenna model {}".format(name))
            else:
                self._open_antenna_patterns[key] = AntennaPattern(name, **kwargs)

        return self._open_antenna_patterns[key]

    def _is_buffered(self, name):
        """
        Check whether any variant of the antenna model ``name`` is already buffered
        """
        return any(key[0] == name for key in self._open_antenna_patterns)
