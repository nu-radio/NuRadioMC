import numpy as np
import json
import os
from NuRadioReco.utilities import units, io_utilities
import gzip
from radiotools import helper as hp
from radiotools import coordinatesystems as cs
from scipy import constants
import logging
import pickle
logger = logging.getLogger('NuRadioReco.antennapattern')

# config = ConfigParser.RawConfigParser()
# config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'antennamodels.cfg')
# if not os.path.exists(config_path):
#     logger.error("antennamodels.cfg does not exists. You need to create this file manually from the antennamodels.cfg.sample file and add the path to the antenna models svn reop")
#     raise OSError
# config.read(config_path)
path_to_antennamodels = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AntennaModels')


def interpolate_linear(x, x0, x1, y0, y1, interpolation_method='complex'):
    """
    helper function to linearly interpolate between two complex numbers

    Parameters:
    ------
    x: float
        the requested position
    x0, y0: float, complex float
        the first data point
    x1, y1: float, complex float
        the second data point
    interpolation_method: string
        specifies if interpolation is in
        * complex (default) i.e. real and imaginary part
        * magnitude and phase

    Returns: compex float
        the interpolated value
    """
    if (x0 == x1):
        return y0
    if(interpolation_method == 'complex'):
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    elif(interpolation_method == 'magphase'):  # interpolate magnitude and phase
        mag0 = np.abs(y0)
        mag1 = np.abs(y1)
        phase0 = np.angle(y0)
        phase1 = np.angle(y1)
        phase0, phase1 = np.unwrap([phase0, phase1])
        mag = mag0 + (mag1 - mag0) * (x - x0) / (x1 - x0)
        phase = phase0 + (phase1 - phase0) * (x - x0) / (x1 - x0)
        y = mag * np.exp(1j * phase)
        return y
    else:
        logger.error("interpolation mode {} not implemented".format(interpolation_method))
        raise NotImplementedError


def interpolate_linear_vectorized(x, x0, x1, y0, y1, interpolation_method='complex'):
    """
    Same as `interpolate_linear` but all parameters can be vectors

    """
    x = np.array(x)
    mask = x0 != x1
    result = np.zeros_like(x, dtype=np.complex)
    denominator = x1 - x0
    if(interpolation_method == 'complex'):
        result[mask] = y0[mask] + (y1[mask] - y0[mask]) * (x[mask] - x0[mask]) / denominator[mask]
    elif(interpolation_method == 'magphase'):  # interpolate magnitude and phase
        mag0 = np.abs(y0[mask])
        mag1 = np.abs(y1[mask])
        phase0 = np.angle(y0[mask])
        phase1 = np.angle(y1[mask])
        phase0, phase1 = np.unwrap([phase0, phase1])
        mag = mag0 + (mag1 - mag0) * (x[mask] - x0[mask]) / denominator[mask]
        phase = phase0 + (phase1 - phase0) * (x[mask] - x0[mask]) / denominator[mask]
        result[mask] = mag * np.exp(1j * phase)
    else:
        logger.error("interpolation mode {} not implemented".format(interpolation_method))
        raise NotImplementedError
    result[~mask] = y0[~mask]
    return result


def get_group_delay(vector_effective_length, df):
    """
    helper function to calculate the group delay from the vector effecitve length

    Parameters:
    ----------
    vector_effective_length: complex float
        the vector effective length of an antenna
    df: float
        the size of a frequency bin

    Returns: float (the group delay)


    """
    return -np.diff(np.unwrap(np.angle(vector_effective_length))) / df / units.ns / 2 / np.pi


def parse_WIPLD_file(ad1, ra1, orientation, gen_num=1, s_paramateres=[1, 1]):
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

    Returns: all parameters of the files
    """
    boresight, tines = np.loadtxt(orientation, delimiter=',')
    orientation_theta, orientation_phi = hp.cartesian_to_spherical(*boresight)
    rotation_theta, rotation_phi = hp.cartesian_to_spherical(*tines)

    ad1_data = np.loadtxt(ad1, comments='>')
    S_1 = ad1_data[:, 1]
    S_2 = ad1_data[:, 2]
    mask = (S_1 == s_paramateres[0]) & (S_2 == s_paramateres[1])
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
            if(line.strip().startswith('>')):
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
        return orientation_theta, orientation_phi, rotation_theta, rotation_phi, ff, Z, S, np.array(ff2), np.deg2rad(np.array(phis)), np.deg2rad(np.array(thetas)), np.array(Ephis), np.array(Ethetas), np.array(gains)


def preprocess_WIPLD_old(path, gen_num=1, s_paramateres=[1, 1]):
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

    Returns:
        * orientation theta: boresight direction (zenith angle, 0deg is the zenith, 180deg is straight down)
        * orientation phi: boresight direction (azimuth angle counting from East counterclockwise)
        * rotation theta: rotation of the antenna, is perpendicular to 'orientation', for LPDAs: vector in plane of tines pointing away from connector
        * rotation phi: rotation of the antenna, is perpendicular to 'orientation', for LPDAs: vector in plane of tines pointing away from connector
        * ff2: array of frequencies
        * theta: zenith angle of inicdent electric field
        * phi: azimuth angle of incident electric field
        * H_phi: the complex vector effective length of the ePhi polarization component
        * H_theta: the complex vector effective length of the eTheta polarization component
    """
    from scipy.interpolate import interp1d
    c = constants.c * units.m / units.s
    Z_0 = 119.9169 * np.pi * units.ohm
    split = os.path.split(os.path.dirname(path))
    name = split[1]
    path = split[0]

    orientation_theta, orientation_phi, rotation_theta, rotation_phi, ff, Z, S, ff2, phi, theta, Iphi, Itheta, gains = parse_WIPLD_file(os.path.join(path, name, '{}.ad1'.format(name)),
                                                                                                                   os.path.join(path, name, '{}.ra1'.format(name)),
                                                                                                                   os.path.join(path, name, '{}.orientation'.format(name)),
                                                                                                                   gen_num=gen_num, s_paramateres=s_paramateres)

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
    H_phi = (2 * wavelength * get_Z(ff2) * Iphi) / (Z_0) / 1j
    H_theta = (2 * wavelength * get_Z(ff2) * Itheta) / (Z_0) / 1j

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
    orientation_theta, orientation_phi, rotation_theta, rotation_phi, ff2, theta, phi, H_phi, H_theta = preprocess_WIPLD_old(path)
    split = os.path.split(os.path.dirname(path))
    name = split[1]
    path = split[0]
    output_filename = '{}.pkl'.format(os.path.join(path, name, name))
    with open(output_filename, 'wb') as fout:
        logger.info('saving output to {}'.format(output_filename))
        pickle.dump([orientation_theta, orientation_phi, rotation_theta, rotation_phi, ff2, theta, phi, H_phi, H_theta], fout, protocol=4)


def preprocess_WIPLD(path, gen_num=1, s_paramateres=[1, 1]):
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

    Returns:
        * orientation theta: boresight direction (zenith angle, 0deg is the zenith, 180deg is straight down)
        * orientation phi: boresight direction (azimuth angle counting from East counterclockwise)
        * rotation theta: rotation of the antenna, is perpendicular to 'orientation', for LPDAs: vector in plane of tines pointing away from connector
        * rotation phi: rotation of the antenna, is perpendicular to 'orientation', for LPDAs: vector in plane of tines pointing away from connector
        * ff2: array of frequencies
        * theta: zenith angle of inicdent electric field
        * phi: azimuth angle of incident electric field
        * H_phi: the complex vector effective length of the ePhi polarization component
        * H_theta: the complex vector effective length of the eTheta polarization component
    """
    from scipy.interpolate import interp1d
    c = constants.c * units.m / units.s
    Z_0 = 119.9169 * np.pi * units.ohm
    split = os.path.split(os.path.dirname(path))
    name = split[1]
    path = split[0]

    orientation_theta, orientation_phi, rotation_theta, rotation_phi, ff, Z, S, ff2, phi, theta, Iphi, Itheta, gains = parse_WIPLD_file(os.path.join(path, name, '{}.ad1'.format(name)),
                                                                                                                   os.path.join(path, name, '{}.ra1'.format(name)),
                                                                                                                   os.path.join(path, name, '{}.orientation'.format(name)),
                                                                                                                   gen_num=gen_num, s_paramateres=s_paramateres)

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
    H_phi = wavelength * (1 + get_S(ff2)) * Iphi * Z_L / (Z_0) / 1j / V
    H_theta = wavelength * (1 + get_S(ff2)) * Itheta * Z_L / (Z_0) / 1j / V

#     H = wavelength * (np.real(get_Z(ff2)) / (np.pi * Z_0)) ** 0.5 * gains ** 0.5
    return orientation_theta, orientation_phi, rotation_theta, rotation_phi, ff2, theta, phi, H_phi, H_theta

#     output_filename = '{}.pkl'.format(os.path.join(path, name, name))
#     with open(output_filename, 'wb') as fout:
#         logger.info('saving output to {}'.format(output_filename))
#         pickle.dump([orientation_theta, orientation_phi, rotation_theta, rotation_phi, ff2, theta, phi, H_phi, H_theta], fout, protocol=4)


def save_preprocessed_WIPLD(path):
    """
    saves preprocessed WIPLD files to a pickle file

    Parameters
    ----------
    path: string
        path to folder containing ad1, ra1, and orientation files.
    """
    orientation_theta, orientation_phi, rotation_theta, rotation_phi, ff2, theta, phi, H_phi, H_theta = preprocess_WIPLD(path)
    split = os.path.split(os.path.dirname(path))
    name = split[1]
    path = split[0]
    output_filename = '{}.pkl'.format(os.path.join(path, name, name))
    with open(output_filename, 'wb') as fout:
        logger.info('saving output to {}'.format(output_filename))
        pickle.dump([orientation_theta, orientation_phi, rotation_theta, rotation_phi, ff2, theta, phi, H_phi, H_theta], fout, protocol=4)


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

    orientation_theta, orientation_phi, rotation_theta, rotation_phi, ff, Z, S, ff2, phi, theta, Iphi, Itheta, gains = parse_WIPLD_file(os.path.join(path, name, '{}.ad1'.format(name)),
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
    H_phi = wavelength * (1 + get_S(ff2)) * Iphi * Z_L / (Z_0) / 1j / V
    H_theta = wavelength * (1 + get_S(ff2)) * Itheta * Z_L / (Z_0) / 1j / V

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


def get_pickle_antenna_response(path):
    """
    opens and return the pickle file containing the preprocessed WIPL-D antenna simulation
    If the pickle file is not present on the local file system, or if the file is outdated (verified via a sha1 hash sum),
    the file will be downloaded from a central data server


    Parameters:
    ----------
    path: string
        the path to the pickle file

    """

    download_file = False

    # check if gziped pickle file already exists
    if(not os.path.exists(path)):
        logger.warning("antenna pattern {} does not exist, file will be downloaded".format(path))
        download_file = True

    if(os.path.exists(path)):
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
            if(os.path.basename(path) in antenna_hashs.keys()):
                if(sha1.hexdigest() != antenna_hashs[os.path.basename(path)]):
                    logger.warning("antenna model {} has changed on the server. downloading newest version...".format(os.path.basename(path)))
                    download_file = True
            else:
                logger.warning("no hash sum of {} available, skipping up-to-date check".format(os.path.basename(path)))

    if(download_file):
        # does not exist yet -> download file
        import requests
        antenna_pattern_name = os.path.splitext(os.path.basename(path))[0]
        URL = 'http://arianna.ps.uci.edu/~arianna/data/AntennaModels/{name}/{name}.pkl'.format(name=antenna_pattern_name)

        folder = os.path.dirname(path)
        if(not os.path.exists(folder)):
            os.makedirs(folder)
        logger.info("downloading antenna pattern {} from {}. This can take a while...".format(antenna_pattern_name, URL))
        r = requests.get(URL)
        if (r.status_code != requests.codes.ok):
            logger.error("error in download of antenna model")
            raise IOError
        with open(path, "wb") as code:
            code.write(r.content)
        logger.warning("...download finished.")

#         # does not exist yet -> precalculating WIPLD simulations from raw WIPLD output
#         preprocess_WIPLD(path)
    res = io_utilities.read_pickle(path, encoding='bytes')
    return res


def parse_AERA_XML_file(path):
    import xml.etree.ElementTree as ET

    if not os.path.exists(path):
        logger.error("AERA antenna file {} not found".format(path))
        raise OSError

    antenna_file = open(path, "rb")

    antenna_data = "<antenna>" + antenna_file.read() + "</antenna>"  # add pseudo root element

    # get root element
    root = ET.fromstring(antenna_data)

    # get frequencies and angles
    frequencies_node = root.find("./frequency")
    frequencies = np.array(frequencies_node.text.strip().split(), dtype=np.float) * units.MHz

    theta_node = root.find("./theta")
    thetas = np.array(theta_node.text.strip().split(), dtype=np.float) * units.deg

    phi_node = root.find("./phi")
    phis = np.array(phi_node.text.strip().split(), dtype=np.float) * units.deg

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
        if(theta_amp_node is None):
            freq_string = "%.1f" % freq

        theta_amp_node = root.find("./EAHTheta_amp[@idfreq='%s']" % freq_string)
        theta_amps[iFreq] = np.array(theta_amp_node.text.strip().split(), dtype=np.float) * units.m

        theta_phase_node = root.find("./EAHTheta_phase[@idfreq='%s']" % freq_string)
        theta_phases[iFreq] = np.deg2rad(np.array(theta_phase_node.text.strip().split(" "), dtype=np.float))

        phi_amp_node = root.find("./EAHPhi_amp[@idfreq='%s']" % freq_string)
        phi_amps[iFreq] = np.array(phi_amp_node.text.strip().split(), dtype=np.float) * units.m

        phi_phase_node = root.find("./EAHPhi_phase[@idfreq='%s']" % freq_string)
        phi_phases[iFreq] = np.deg2rad(np.array(phi_phase_node.text.strip().split(), dtype=np.float))

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
    output_filename = '{}_InfAir.pkl'.format(os.path.join(path_to_antennamodels, fname, fname))

    directory = os.path.dirname(output_filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(output_filename, 'wb') as fout:
        logger.info('saving output to {}'.format(output_filename))
        pickle.dump([orientation_theta, orientation_phi, rotation_theta, rotation_phi, ff, theta, phi, H_phi, H_theta], fout, protocol=4)


def parse_ARA_file(ara):
    """
    Helper function that parses the ARAsim ASCII files containig antenna responses

    Parameters:
    ----------
    ara: string
        path to the file

    Returns:
        * ff: array of floats
            frequencies
        * thetas: array of floats
            zenith angle of inicdent electric field
        * phis: array of floats
            azimuth angle of inicdent electric field
        * gains: array of floats
            corresponding linear gain values
        * phases: array of floats
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
            if(line.strip().startswith('freq')):
                # add phi = 360deg = 0deg to data structure (to allow for a automated interpolation
                f = float(line.replace(" ", "").replace("freq", "").replace(":", "").replace("MHz", ""))
                continue
            if(line.strip().startswith('SWR')):
                continue
            if(line.strip().startswith('Theta')):
                continue
            ff.append(f * units.MHz)
            theta, phi, gaindB, gain, phase = line.split()
            if(float(phi) == 0):
                tmp_phi0_lines.append(line)
            phis.append(float(phi) * units.deg)
            thetas.append(float(theta) * units.deg)
            gains.append(float(gain))
            phases.append(float(phase) * units.deg)
            if(float(phi) == 355 and float(theta) == 180):
                for i, tline in enumerate(tmp_phi0_lines):
                    ff.append(f * units.MHz)
                    theta, phi, gaindB, gain, phase = tline.split()
                    if(i == 0):
                        logger.debug("{} {} {} {} {} {}".format(f, theta, phi, gaindB, gain, phase))
                    phis.append(360. * units.deg)
                    thetas.append(float(theta) * units.deg)
                    gains.append(float(gain))
                    phases.append(float(phase) * units.deg)
                tmp_phi0_lines = []

        return np.array(ff), np.array(phis), np.array(thetas), np.array(gains), np.array(phases)


def preprocess_ARA(path):
    """
    preprocess an antenna pattern in the ARASim ASCII file format. The vector effective length is calculated and
    the output is saved to the NuRadioReco pickle format.

    Parameters:
    ----------
    path: string
        the path to the file

    """
    c = constants.c * units.m / units.s
    Z_0 = 119.9169 * np.pi
    split = os.path.split(os.path.dirname(path))
    name = split[1]
    path = split[0]
    orientation_theta, orientation_phi, rotation_theta, rotation_phi, ff, phi, theta, gain, phase = parse_ARA_file(os.path.join(path, name, '{}.txt'.format(name)),
                                                                                                 os.path.join(path, name, '{}.orientation'.format(name)))

    wavelength = c / ff
    H_theta = wavelength * (50 / (np.pi * Z_0)) ** 0.5 * gain ** 0.5  # * np.exp(1j * phase)  ## do not use phases, this will screw up the interpolation
    H_phi = H_theta * 1e-3
    output_filename = '{}.pkl'.format(os.path.join(path, name, name))
    with open(output_filename, 'wb') as fout:
        logger.info('saving output to {}'.format(output_filename))
        pickle.dump([orientation_theta, orientation_phi, rotation_theta, rotation_phi, ff, theta, phi, H_phi, H_theta], fout, protocol=4)


def parse_HFSS_file(hfss):
    """
    Helper function that parses the HFSS files containig antenna responses

    Parameters:
    ----------
    hfss: string
        path to the file

    Returns:
        * ff: array of floats
            frequencies
        * thetas: array of floats
            zenith angle of inicdent electric field
        * phis: array of floats
            azimuth angle of inicdent electric field
        * magnitudes_theta: array of floats
            corresponding logarithmic magnitude values theta component
        * magnitudes_phi: array of floats
            corresponding logarithmic magnitude values phi component
        * phases_phi: array of floats
            corresponding phases phi component
        * phases_theta: array of floats
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
                arr[(i - 1) * len(ff) / len(np.unique(ff)):i * len(ff) / len(np.unique(ff))] = [x for _, x in sorted(zip(phi[(i - 1) * len(ff) / len(np.unique(ff)):i * len(ff) / len(np.unique(ff))], arr[(i - 1) * len(ff) / len(np.unique(ff)):i * len(ff) / len(np.unique(ff))]), key=lambda pair: pair[0])]

        return np.array(ff), np.array(phi), np.array(theta), np.array(mag_phi), np.array(mag_theta), np.array(phase_phi), np.array(phase_theta)


def preprocess_HFSS(path):

    """
    preprocess an antenna pattern in the HFSS file format. The vector effective length is calculated and the output is saved in the NuRadioReco pickle format.

    The vector effective length calculation still needs to be verified.

    The frequencies, theta, phi, magnitude theta, magnitude phi, phase theta and phase phi are read from the csv file and than ordered according to the NuRadioReco format.


    Parameters:
    ----------
    path: string
        the path to the file

    """

    split = os.path.split(os.path.dirname(path))
    name = split[1]
    path = split[0]

    ff, phi, theta, mag_phi, mag_theta, phase_phi, phase_theta = parse_HFSS_file((os.path.join(path, name, '{}.csv'.format(name))))
    mag_theta = 10 ** (mag_theta / 10)
    mag_phi = 10 ** (mag_phi / 10)
    gain_theta = 4.0 * np.pi * (mag_theta ** 2) / (2 * 120 * np.pi)
    gain_phi = 4.0 * np.pi * (mag_phi ** 2) / (2 * 120 * np.pi)
    c = constants.c * units.m / units.s
    Z_0 = 119.9169 * np.pi
    wavelength = c / np.array(ff)
    n_index = 1.78

    H_theta = wavelength / n_index ** 0.5 * (50 / (4 * np.pi * Z_0)) ** 0.5 * gain_theta ** 0.5 * np.exp(1j * phase_theta)
    H_phi = wavelength / n_index ** 0.5 * (50 / (4 * np.pi * Z_0)) ** 0.5 * gain_phi ** 0.5 * np.exp(1j * phase_phi)

    orientation_theta = 0
    orientation_phi = 0
    rotation_theta = 0
    rotation_phi = 0

    output_filename = '{}.pkl'.format(os.path.join(path, name, name))

    with open(output_filename, 'wb') as fout:
        logger.info('saving output to {}'.format(output_filename))
        pickle.dump([orientation_theta, orientation_phi, rotation_theta, rotation_phi, ff, theta, phi, H_phi, H_theta], fout, protocol=4)


def preprocess_XFDTD(path):
    """
    preprocess an antenna pattern in the XFDTD file format. The vector effective length is calculated and
    the output is saved to the NuRadioReco pickle format.

    Parameters:
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
        if(info['type'] == 'Vpol'):
            H_theta = H
            H_phi = H * 1e-6
        elif(info['type'] == 'Hpol'):
            H_theta = H * 1e-6
            H_phi = H
        else:
            logger.error("antenna type {} not understood".format(info['type']))
            raise NotImplementedError("antenna type {} not understood".format(info['type']))

        output_filename = '{}.pkl'.format(os.path.join(path, name, name))
        with open(output_filename, 'wb') as fout:
            logger.info('saving output to {}'.format(output_filename))
            pickle.dump([orientation_theta, orientation_phi, rotation_theta, rotation_phi, ff, theta, phi, H_phi, H_theta], fout, protocol=4)


class AntennaPatternBase():
    """
    base class of utility class that handles access and buffering to antenna pattern
    """

    def _get_antenna_rotation(self, orientation_theta, orientation_phi, rotation_theta, rotation_phi):
        """

        Parameters:
        ----------

        """
        # define orientation of wiplD antenna simulation (in ARIANNA CS)
        e1 = hp.spherical_to_cartesian(self._orientation_theta, self._orientation_phi)  # boresight direction
        e2 = hp.spherical_to_cartesian(self._rotation_theta, self._rotation_phi)  # vector perpendicular to tine plane
        e3 = np.cross(e1, e2)
        E = np.array([e1, e2, e3])
        if(np.linalg.norm(e3) < 0.9):
            logger.error("orientation of antenna not properly defined in WIPL-D orientation file")
            raise AssertionError("orientation of antenna not properly defined in WIPL-D orientation file")

        # get normal vectors for antenne orientation in field (in ARIANNA CS)
        a1 = hp.spherical_to_cartesian(orientation_theta, orientation_phi)
        a2 = hp.spherical_to_cartesian(rotation_theta, rotation_phi)
        a3 = np.cross(a1, a2)
        A = np.array([a1, a2, a3])
        if(np.linalg.norm(a3) < 0.9):
            logger.error("orientation of antenna not properly defined detector description")
            raise AssertionError("orientation of antenna not properly defined detector description")
        from numpy.linalg import inv

        return np.matmul(inv(E), A)

    def _get_theta_and_phi(self, zenith, azimuth, orientation_theta, orientation_phi, rotation_theta, rotation_phi):
        """
        transform zenith and azimuth angle in ARIANNA coordinate system to the WIPLD coordinate system.
        In addition the orientation of the antenna as deployed in the field is taken into account.

        Parameters:
        ----------
        """

        rot = self._get_antenna_rotation(orientation_theta, orientation_phi, rotation_theta, rotation_phi)

        incoming_direction = hp.spherical_to_cartesian(zenith, azimuth)
        incoming_direction_WIPLD = np.dot(rot, incoming_direction.T).T
        theta, phi = hp.cartesian_to_spherical(*incoming_direction_WIPLD)
        if(zenith == 180 * units.deg):
            logger.debug(incoming_direction)
            logger.debug(rot)
            logger.debug(incoming_direction_WIPLD)
#         theta = 0.5 * np.pi - theta  # in wipl D the elevation is defined with 0deg being in the x-y plane
#         theta = hp.get_normalized_angle(theta)
#         phi = hp.get_normalized_angle(phi)

        logger.debug("zen/az {:.0f} {:.0f} transform to {:.0f} {:.0f}".format(zenith / units.deg,
                                                                              azimuth / units.deg,
                                                                              theta / units.deg,
                                                                              phi / units.deg))
        return theta, phi

    def get_antenna_response_vectorized(self, freq, zenith, azimuth, orientation_theta, orientation_phi, rotation_theta, rotation_phi):
        """
        get the antenna response for a specific frequency, zenith and azimuth angle

        All angles are specified in the ARIANNA coordinate system. All units are in ARIANNA default units

        Parameters
        ----------
        freq : float or array of floats
            frequency
        zenith : float
            zenith angle of incoming signal direction
        azimuth : float
            azimuth angle of incoming signal direction
        orientation_theta: float 
            boresight direction (zenith angle, 0deg is the zenith, 180deg is straight down)
        orientation_phi: float 
            boresight direction (azimuth angle counting from East counterclockwise)
        rotation_theta: float
            rotation of the antenna, is perpendicular to 'orientation', for LPDAs: vector in plane of tines pointing away from connector
        rotation_phi: float 
            rotation of the antenna, is perpendicular to 'orientation', for LPDAs: vector in plane of tines pointing away from connector

        Returns
        -------
        VEL: dictonary of complex arrays
            theta and phi component of the vector effective length, both components
            are complex floats or arrays of complex floats
            of the same length as the frequency input
        """
        if self._notfound:
            VEL = {}
            VEL['theta'] = np.ones(len(freq), dtype=np.complex)
            VEL['phi'] = np.ones(len(freq), dtype=np.complex)
            return VEL

        if(isinstance(freq, (float, int))):
            freq = np.array([freq])
        theta, phi = self._get_theta_and_phi(zenith, azimuth, orientation_theta, orientation_phi, rotation_theta, rotation_phi)

        Vtheta_raw, Vphi_raw = self._get_antenna_response_vectorized_raw(freq, theta, phi)

        # now rotate the raw theta and phi component of the VEL into the ARIANNA coordinate system.
        # As the theta and phi angles are differently defined in WIPLD and ARIANNA, also the orientation of the
        # eTheta and ePhi unit vectors are different.
        cstrans = cs.cstrafo(zenith=theta, azimuth=phi)
        V_xyz_raw = cstrans.transform_from_onsky_to_ground(np.array([np.zeros(Vtheta_raw.shape[0]), Vtheta_raw, Vphi_raw]))
        rot = self._get_antenna_rotation(orientation_theta, orientation_phi, rotation_theta, rotation_phi)
        from numpy.linalg import inv
        V_xyz = np.dot(inv(rot), V_xyz_raw)

        cstrans2 = cs.cstrafo(zenith=zenith, azimuth=azimuth)
        V_onsky = cstrans2.transform_from_ground_to_onsky(V_xyz)
        VEL = {}
        VEL['theta'] = V_onsky[1]
        VEL['phi'] = V_onsky[2]
        return VEL


class AntennaPattern(AntennaPatternBase):
    """
    utility class that handles access and buffering to simulated antenna pattern
    """

    def __init__(self, antenna_model, path=path_to_antennamodels,
                 interpolation_method='complex'):
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
        """

        self._name = antenna_model
        self._interpolation_method = interpolation_method
        from time import time
        t = time()
        filename = os.path.join(path, antenna_model, "{}.pkl".format(antenna_model))
        self._notfound = False
        try:
            self._orientation_theta, self._orientation_phi, self._rotation_theta, self._rotation_phi, \
                    ff, thetas, phis, H_phi, H_theta = get_pickle_antenna_response(filename)

        except IOError:
            self._notfound = True
            logger.warning("antenna response for {} not found".format(antenna_model))
            return

        self.frequencies = np.unique(ff)
        self.frequency_lower_bound = self.frequencies[0]
        self.frequency_upper_bound = self.frequencies[-1]

        self.theta_angles = np.unique(thetas)
        self.theta_lower_bound = self.theta_angles[0]
        self.theta_upper_bound = self.theta_angles[-1]
        logger.debug("{} thetas from {} to {}".format(len(self.theta_angles), self.theta_lower_bound, self.theta_upper_bound))

        self.phi_angles = np.unique(phis)
        self.phi_lower_bound = self.phi_angles[0]
        self.phi_upper_bound = self.phi_angles[-1]
        logger.debug("{} phis from {} to {}".format(len(self.phi_angles), self.phi_lower_bound, self.phi_upper_bound))

        self.n_freqs = len(self.frequencies)
        self.n_theta = len(self.theta_angles)
        self.n_phi = len(self.phi_angles)

        self.VEL_phi = H_phi
        self.VEL_theta = H_theta

        # additional consistency check
        for iFreq, freq in enumerate(self.frequencies):
            for iPhi, phi in enumerate(self.phi_angles):
                for iTheta, theta in enumerate(self.theta_angles):
                    index = self._get_index(iFreq, iTheta, iPhi)

                    if (phi != phis[index]):
                        logger.error("phi angle has changed during theta loop {0}, {1}".format(
                                                phi / units.deg, phis[index] / units.deg))
                        raise Exception("phi angle has changed during theta loop")

                    if (theta != thetas[index]):
                        logger.error("theta angle has changed during theta loop {0}, {1}".format(
                                                theta / units.deg, thetas[index] / units.deg))
                        raise Exception("theta angle has changed during theta loop")

                    if (freq != ff[index]):
                        logger.error("frequency has changed {0}, {1}".format(
                                                        freq, ff[index]))
                        raise Exception("frequency has changed")

        logger.warning('loading antenna file {} took {:.0f} seconds'.format(antenna_model, time() - t))

    def _get_index(self, iFreq, iTheta, iPhi):
        """
        """
        return iFreq * self.n_theta * self.n_phi + iPhi * self.n_theta + iTheta

    def _get_antenna_response_vectorized_raw(self, freq, theta, phi):
        """
        get vector effective length in WIPLD coordinate system
        """
        while phi < self.phi_lower_bound:
            phi += 2 * np.pi
        while phi > self.phi_upper_bound:
            phi -= 2 * np.pi

        if(hp.is_equal(theta, self.theta_upper_bound, rel_precision=1e-5)):
            theta = self.theta_upper_bound
        if(hp.is_equal(theta, self.theta_lower_bound, rel_precision=1e-5)):
            theta = self.theta_lower_bound
        if(((phi < self.phi_lower_bound) or (phi > self.phi_upper_bound)) or
           ((theta < self.theta_lower_bound) or (theta > self.theta_upper_bound))):
            logger.debug(self._name)
            logger.debug("theta bounds {0} ,{1}, {2}".format(self.theta_lower_bound, theta, self.theta_upper_bound))
            logger.debug("phi bounds {0} ,{1}, {2}".format(self.phi_lower_bound, phi, self.phi_upper_bound))
            logger.warning("theta, phi or frequency out of range, returning (0,0j)")
            logger.debug("{0},{1},{2}".format(freq, self.frequency_lower_bound, self.frequency_upper_bound))
            return (0, 0)

        if(self.theta_upper_bound == self.theta_lower_bound):
            iTheta_lower = 0
            iTheta_upper = 0
        else:
            iTheta_lower = np.array(np.floor((theta - self.theta_lower_bound) / (self.theta_upper_bound - self.theta_lower_bound) * (self.n_theta - 1)), dtype=np.int)
            iTheta_upper = np.array(np.ceil((theta - self.theta_lower_bound) / (self.theta_upper_bound - self.theta_lower_bound) * (self.n_theta - 1)), dtype=np.int)
        theta_lower = self.theta_angles[iTheta_lower]
        theta_upper = self.theta_angles[iTheta_upper]
        if(self.phi_upper_bound == self.phi_lower_bound):
            iPhi_lower = 0
            iPhi_upper = 0
        else:
            iPhi_lower = np.array(np.floor((phi - self.phi_lower_bound) / (self.phi_upper_bound - self.phi_lower_bound) * (self.n_phi - 1)), dtype=np.int)
            iPhi_upper = np.array(np.ceil((phi - self.phi_lower_bound) / (self.phi_upper_bound - self.phi_lower_bound) * (self.n_phi - 1)), dtype=np.int)
        phi_lower = self.phi_angles[iPhi_lower]
        phi_upper = self.phi_angles[iPhi_upper]

        iFrequency_lower = np.array(np.floor((freq - self.frequency_lower_bound) / (self.frequency_upper_bound - self.frequency_lower_bound) * (self.n_freqs - 1)), dtype=np.int)
        iFrequency_upper = np.array(np.ceil((freq - self.frequency_lower_bound) / (self.frequency_upper_bound - self.frequency_lower_bound) * (self.n_freqs - 1)), dtype=np.int)
        # handling frequency out of bound cases properly
        out_of_bound_freqs_low = freq < self.frequency_lower_bound
        out_of_bound_freqs_high = freq > self.frequency_upper_bound
        iFrequency_lower[out_of_bound_freqs_low] = 0  # set all out of bound frequencies to its minimum/maximum possible value
        iFrequency_lower[out_of_bound_freqs_high] = 0  # set all out of bound frequencies to its minimum/maximum possible value
        iFrequency_upper[out_of_bound_freqs_low] = self.n_freqs - 1
        iFrequency_upper[out_of_bound_freqs_high] = self.n_freqs - 1
        frequency_lower = self.frequencies[iFrequency_lower]
        frequency_upper = self.frequencies[iFrequency_upper]

        # lower frequency bound
        # theta low
        VELt_freq_low_theta_low = interpolate_linear(
            phi, phi_lower, phi_upper,
            self.VEL_theta[self._get_index(iFrequency_lower, iTheta_lower, iPhi_lower)],
            self.VEL_theta[self._get_index(iFrequency_lower, iTheta_lower, iPhi_upper)],
            self._interpolation_method)
        VELp_freq_low_theta_low = interpolate_linear(
            phi, phi_lower, phi_upper,
            self.VEL_phi[self._get_index(iFrequency_lower, iTheta_lower, iPhi_lower)],
            self.VEL_phi[self._get_index(iFrequency_lower, iTheta_lower, iPhi_upper)],
            self._interpolation_method)

        # theta up
        VELt_freq_low_theta_up = interpolate_linear(
            phi, phi_lower, phi_upper,
            self.VEL_theta[self._get_index(iFrequency_lower, iTheta_upper, iPhi_lower)],
            self.VEL_theta[self._get_index(iFrequency_lower, iTheta_upper, iPhi_upper)],
            self._interpolation_method)
        VELp_freq_low_theta_up = interpolate_linear(
            phi, phi_lower, phi_upper,
            self.VEL_phi[self._get_index(iFrequency_lower, iTheta_upper, iPhi_lower)],
            self.VEL_phi[self._get_index(iFrequency_lower, iTheta_upper, iPhi_upper)],
            self._interpolation_method)

        VELt_freq_low = interpolate_linear(theta, theta_lower,
                                           theta_upper,
                                           VELt_freq_low_theta_low,
                                           VELt_freq_low_theta_up,
                                           self._interpolation_method)
        VELp_freq_low = interpolate_linear(theta, theta_lower,
                                           theta_upper,
                                           VELp_freq_low_theta_low,
                                           VELp_freq_low_theta_up,
                                           self._interpolation_method)

        # upper frequency bound
        # theta low
        VELt_freq_up_theta_low = interpolate_linear(
            phi, phi_lower, phi_upper,
            self.VEL_theta[self._get_index(iFrequency_upper, iTheta_lower, iPhi_lower)],
            self.VEL_theta[self._get_index(iFrequency_upper, iTheta_lower, iPhi_upper)],
            self._interpolation_method)
        VELp_freq_up_theta_low = interpolate_linear(
            phi, phi_lower, phi_upper,
            self.VEL_phi[self._get_index(iFrequency_upper, iTheta_lower, iPhi_lower)],
            self.VEL_phi[self._get_index(iFrequency_upper, iTheta_lower, iPhi_upper)],
            self._interpolation_method)

        # theta up
        VELt_freq_up_theta_up = interpolate_linear(
            phi, phi_lower, phi_upper,
            self.VEL_theta[self._get_index(iFrequency_upper, iTheta_upper, iPhi_lower)],
            self.VEL_theta[self._get_index(iFrequency_upper, iTheta_upper, iPhi_upper)],
            self._interpolation_method)
        VELp_freq_up_theta_up = interpolate_linear(
            phi, phi_lower, phi_upper,
            self.VEL_phi[self._get_index(iFrequency_upper, iTheta_upper, iPhi_lower)],
            self.VEL_phi[self._get_index(iFrequency_upper, iTheta_upper, iPhi_upper)],
            self._interpolation_method)

        VELt_freq_up = interpolate_linear(theta, theta_lower, theta_upper,
                                          VELt_freq_up_theta_low,
                                          VELt_freq_up_theta_up,
                                          self._interpolation_method)
        VELp_freq_up = interpolate_linear(theta, theta_lower, theta_upper,
                                          VELp_freq_up_theta_low,
                                          VELp_freq_up_theta_up,
                                          self._interpolation_method)

        interpolated_VELt = interpolate_linear_vectorized(freq, frequency_lower,
                                                          frequency_upper,
                                                          VELt_freq_low,
                                                          VELt_freq_up,
                                                          self._interpolation_method)
        interpolated_VELp = interpolate_linear_vectorized(freq, frequency_lower,
                                                          frequency_upper,
                                                          VELp_freq_low,
                                                          VELp_freq_up,
                                                          self._interpolation_method)

        # set all out of bound frequencies to zero
        interpolated_VELt[out_of_bound_freqs_low] = 0 + 0 * 1j
        interpolated_VELt[out_of_bound_freqs_high] = 0 + 0 * 1j
        interpolated_VELp[out_of_bound_freqs_low] = 0 + 0 * 1j
        interpolated_VELp[out_of_bound_freqs_high] = 0 + 0 * 1j
        return interpolated_VELt, interpolated_VELp


class AntennaPatternAnalytic(AntennaPatternBase):
    """
    utility class that handles access and buffering to analytic antenna pattern
    """

    def __init__(self, antenna_model, cutoff_freq=50 * units.MHz):
        """

        """
        self._notfound = False
        self._model = antenna_model
        self._cutoff_freq = cutoff_freq
        if(self._model == 'analytic_LPDA'):
            # LPDA dummy model points towards z direction and has its tines in the y-z plane
            logger.info("setting boresight direction")
            self._orientation_theta = 0 * units.deg
            self._orientation_phi = 0 * units.deg
            self._rotation_theta = 90 * units.deg
            self._rotation_phi = 0 * units.deg

    def parametric_phase(self, freq, type='theoretical'):
        """

        """
        if type == 'frontlobe_lpda':
            a = 100 * (freq - 400 * units.MHz) ** 2 - 20
            a[np.where(freq > 400 * units.MHz)] -= 0.00007 * (freq[np.where(freq > 400 * units.MHz)] - 400 * units.MHz) ** 2
        elif type == 'side_lpda':
            a = 40 * (freq - 950 * units.MHz) ** 2 - 40
        elif type == 'back_lpda':
            a = 50 * (freq - 950 * units.MHz) ** 2 - 50
        elif type == "theoretical":
            # ratio of two elements
            tau = 0.75
            # maximum frequency
            f = 1000. * units.MHz
            a = np.pi / np.log(tau) * np.log(freq / f) - 60

        return a

    def _get_antenna_response_vectorized_raw(self, freq, theta, phi, group_delay='frontlobe_lpda'):
        """

        """
        if(self._model == 'analytic_LPDA'):
            """
            Dummy LPDA model.
            Flat gain as function of frequency, no group delay.
            Can be used instead of __get_antenna_response_vectorized_raw
            """
            max_gain_co = 4
            max_gain_cross = 2  # Check whether these values are actually reasonable

            index = np.argmax(freq > self._cutoff_freq)
            Gain = np.ones_like(freq)
            from scipy.signal import hann
            filter = hann(2 * index)
            Gain[:index] = filter[:index]

            # at WIPL-D (1,0,0) Gain max for e_theta (?? I hope)
            # Standard units, deliver H_eff in meters
            Z_0 = constants.physical_constants['characteristic impedance of vacuum'][0] * units.ohm
            Z_ant = 50 * units.ohm

            # Assuming simple cosine, sine falls-off for dummy module
            H_eff_t = np.zeros_like(Gain)
            fmask = freq >= 0
            H_eff_t[fmask] = Gain[fmask] * max_gain_cross * 1 / freq[fmask]
            H_eff_t *= np.cos(theta) * np.sin(phi)
            H_eff_t *= constants.c * units.m / units.s * Z_ant / Z_0 / np.pi

            H_eff_p = np.zeros_like(Gain)
            H_eff_p[fmask] = Gain[fmask] * max_gain_co * 1 / freq[fmask]
            H_eff_p *= np.cos(phi)
            H_eff_p *= constants.c * units.m / units.s * Z_ant / Z_0 / np.pi

            if group_delay != None:
                # add here antenna model with analytic description of typical group delay
                phase = self.parametric_phase(freq, group_delay)

                H_eff_p = H_eff_p.astype(complex)
                H_eff_t = H_eff_t.astype(complex)

                H_eff_p *= np.exp(1j * phase)
                H_eff_t *= np.exp(1j * phase)

            return H_eff_p, H_eff_t


class AntennaPatternProvider(object):
    __instance = None

    def __new__(cls):
        if AntennaPatternProvider.__instance is None:
            AntennaPatternProvider.__instance = object.__new__(cls)
        return AntennaPatternProvider.__instance

    def __init__(self):
        """
        Provider class for antenna pattern. The usage of antenna pattern through this class ensures
        that an antenna pattern is loaded only once into memory which takes a significant time and occupies a
        significant amount of memory.
        """
        self._open_antenna_patterns = {}
        self._antenna_model_replacements = {}

        antenna_directory = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(antenna_directory, 'antenna_model_replacements.json')
        if(os.path.exists(filename)):
            with open(filename, 'r') as fin:
                self._antenna_model_replacements = json.load(fin)

    def load_antenna_pattern(self, name, **kwargs):
        """
        loads an antenna pattern and returns the antenna pattern class

        Paramters
        ----------
        name: string
            the name of the antenna pattern
        **kwargs: dict
            key word arguments that are passed to the init function of the `AntennaPattern` class (see
            documentation of this class for further information)
        """
        if(name in self._antenna_model_replacements.keys()):
            if(self._antenna_model_replacements[name] not in self._open_antenna_patterns.keys()):
                logger.warning("local replacement of antenna model requsted: replacing {} with {}".format(name, self._antenna_model_replacements[name]))
            name = self._antenna_model_replacements[name]
        if (name not in self._open_antenna_patterns.keys()):
            if(name.startswith("analytic")):
                self._open_antenna_patterns[name] = AntennaPatternAnalytic(name, **kwargs)
                logger.info("loading analytic antenna model {}".format(name))
            else:
                self._open_antenna_patterns[name] = AntennaPattern(name, **kwargs)
        return self._open_antenna_patterns[name]
