import numpy as np
import json
import os
from NuRadioReco.utilities import units
import gzip
import cPickle as pickle
from radiotools import helper as hp
from radiotools import coordinatesystems as cs
from scipy import constants
import logging
logger = logging.getLogger('antennapattern')
import ConfigParser

# config = ConfigParser.RawConfigParser()
# config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'antennamodels.cfg')
# if not os.path.exists(config_path):
#     logger.error("antennamodels.cfg does not exists. You need to create this file manually from the antennamodels.cfg.sample file and add the path to the antenna models svn reop")
#     raise OSError
# config.read(config_path)
path_to_antennamodels = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AntennaModels')


def interpolate_linear(x, x0, x1, y0, y1):
    if (x0 == x1):
        return y0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def interpolate_linear_vectorized(x, x0, x1, y0, y1):
    x = np.array(x)
    mask = x0 != x1
    result = np.zeros_like(x, dtype=np.complex)
    denominator = x1 - x0
    result[mask] = y0[mask] + (y1[mask] - y0[mask]) * (x[mask] - x0[mask]) / denominator[mask]
    result[~mask] = y0[~mask]
    return result

# def interpolate_linear_VEL(x, x0, x1, y0, y1):
#     result = {}
#     result['theta'] = interpolate_linear(x, x0, x1, y0['theta'], y1['theta'])
#     result['phi'] = interpolate_linear(x, x0, x1, y0['phi'], y1['phi'])
#     return result


def parse_WIPLD_file(ad1, ra1, orientation):
    boresight, tines = np.loadtxt(orientation, delimiter=',')
    zen_boresight, azi_boresight = hp.cartesian_to_spherical(*boresight)
    zen_ori, azi_ori = hp.cartesian_to_spherical(*tines)

    ad1_data = np.loadtxt(ad1, comments='>')
    ff = ad1_data[:, 0] * units.GHz
    Re_Z = ad1_data[:, 5]
    Im_Z = ad1_data[:, 6]
    Z = Re_Z + 1j * Im_Z
    with open(ra1, 'r') as fin:
        ff2 = []
        phis = []
        thetas = []
        Ephis = []
        Ethetas = []
        gains = []
        f = None
        for line in fin.readlines():
#             print(line)
            if(line.strip().startswith('>')):
                f = float(line.split()[4])
#                 print(f)
            else:
                ff2.append(f * units.GHz)
                phi, theta, ReEphi, ImEphi, ReEtheta, ImEtheta, gain, gaindb = line.split()
                phis.append(float(phi))
                thetas.append(float(theta))
                Ephis.append(float(ReEphi) + 1j * float(ImEphi))
                Ethetas.append(float(ReEtheta) + 1j * float(ImEtheta))
                gains.append(float(gain))

#                 print("{:.0f} {:.0f}".format(float(phi), float(theta)))
        if not np.array_equal(ff, np.unique(np.array(ff2))):
            logger.error("error in parsing WIPLD simulation, frequencies of ad1 and ra1 files do not match!")
            return None
        print(np.unique(np.array(phis)))
        print(np.unique(np.array(thetas)))
        return zen_boresight, azi_boresight, zen_ori, azi_ori, ff, Z, np.array(ff2), np.deg2rad(np.array(phis)), np.deg2rad(np.array(thetas)), np.array(Ephis), np.array(Ethetas), np.array(gains)


def preprocess_WIPLD(path):
    from scipy import constants
    from scipy.interpolate import interp1d
    c = constants.c * units.m / units.s
    Z_0 = 119.9169 * np.pi
    split = os.path.split(os.path.dirname(path))
    name = split[1]
    path = split[0]
    zen_boresight, azi_boresight, zen_ori, azi_ori, ff, Z, ff2, phi, theta, Ephi, Etheta, gains = parse_WIPLD_file(os.path.join(path, name, '{}.ad1'.format(name)),
                                                                                                                   os.path.join(path, name, '{}.ra1'.format(name)),
                                                                                                                   os.path.join(path, name, '{}.orientation'.format(name)))

    theta = 0.5 * np.pi - theta  # 90deg - theta because in WIPL D the theta angle is defined differently

    # sort with increasing frequency, increasing phi, and increasing theta
    index = np.lexsort((theta, phi, ff2))
    ff2 = ff2[index]
    phi = phi[index]
    theta = theta[index]
    Ephi = Ephi[index]
    Etheta = Etheta[index]

    get_Z = interp1d(ff, Z, kind='nearest')
    wavelength = c / ff2
    H_phi = (2 * wavelength * get_Z(ff2) * Ephi) / (Z_0) / 1j
    H_theta = (2 * wavelength * get_Z(ff2) * Etheta) / (Z_0) / 1j

#     H = wavelength * (np.real(get_Z(ff2)) / (np.pi * Z_0)) ** 0.5 * gains ** 0.5

    output_filename = '{}.pkl'.format(os.path.join(path, name, name))
    with open(output_filename, 'wb') as fout:
        logger.info('saving output to {}'.format(output_filename))
        pickle.dump([zen_boresight, azi_boresight, zen_ori, azi_ori, ff2, theta, phi, H_phi, H_theta], fout, protocol=2)

#     output_filename = '{}_fromgain.pkl'.format(os.path.join(path, name, name))
#     with open(output_filename, 'wb') as fout:
#         logger.info('saving output to {}'.format(output_filename))
#         pickle.dump([zen_boresight, azi_boresight, zen_ori, azi_ori, ff2, theta, phi, H, np.zeros_like(H)], fout, protocol=2)


def get_WIPLD_antenna_response(path):

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
        print("downloading antenna pattern {} from {}. This can take a while...".format(antenna_pattern_name, URL))
        r = requests.get(URL)
        if (r.status_code != requests.codes.ok):
            print("error in download")
            raise IOError
        with open(path, "wb") as code:
            code.write(r.content)
        print("...download finished.")

#         # does not exist yet -> precalculating WIPLD simulations from raw WIPLD output
#         preprocess_WIPLD(path)
    with open(path, 'rb') as fin:
        res = pickle.load(fin)
        return res


def parse_ARA_file(ara, orientation):
    boresight, tines = np.loadtxt(orientation, delimiter=',')
    zen_boresight, azi_boresight = hp.cartesian_to_spherical(*boresight)
    zen_ori, azi_ori = hp.cartesian_to_spherical(*tines)

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
                        print(f, theta, phi, gaindB, gain, phase)
                    phis.append(360. * units.deg)
                    thetas.append(float(theta) * units.deg)
                    gains.append(float(gain))
                    phases.append(float(phase) * units.deg)
                tmp_phi0_lines = []

#         # reorganize arrays: f, theta, phi -> f, phi, theta
#         nf = len(np.unique(ff))
#         nphi = len(np.unique(np.round(phis, 4)))
#         ntheta = len(np.unique(np.round(thetas, 4)))
#         ff2 = np.zeros_like(ff)
#         phis2 = np.zeros_like(phis)
#         thetas2 = np.zeros_like(thetas)
#         gains2 = np.zeros_like(gains)
#         phases2 = np.zeros_like(phases)
#         for old_index in range(len(ff)):
#             iF = old_index // (nphi * ntheta)
#             iTheta = (old_index - iF * nphi * ntheta) // (nphi)
#             iPhi = old_index - iF * (nphi * ntheta) - iTheta * nphi
#             new_index = iF * nphi * ntheta + iPhi * ntheta + iTheta
#             ff2[new_index] = ff[old_index]
#             phis2[new_index] = phis[old_index]
#             thetas2[new_index] = thetas[old_index]
#             gains2[new_index] = gains[old_index]
#             phases2[new_index] = phases[old_index]
# #
#         iF = 10
#         iPhi = 0
#         for iTheta in range(30):
#             new_index = iF * nphi * ntheta + iPhi * ntheta + iTheta
#             print(ff2[new_index], np.rad2deg(thetas2[new_index]), np.rad2deg(phis2[new_index]), gains2[new_index])

#         return zen_boresight, azi_boresight, zen_ori, azi_ori, ff2, phis2, thetas2, gains2, phases2
        return zen_boresight, azi_boresight, zen_ori, azi_ori, np.array(ff), np.array(phis), np.array(thetas), np.array(gains), np.array(phases)


def preprocess_ARA(path):
    from scipy import constants
    c = constants.c * units.m / units.s
    Z_0 = 119.9169 * np.pi
    split = os.path.split(os.path.dirname(path))
    name = split[1]
    path = split[0]
    zen_boresight, azi_boresight, zen_ori, azi_ori, ff, phi, theta, gain, phase = parse_ARA_file(os.path.join(path, name, '{}.txt'.format(name)),
                                                                                                 os.path.join(path, name, '{}.orientation'.format(name)))
#     print(np.unique(np.rad2deg(phi)))
#     for p in np.unique(phi):
#         print(np.sum(phi == p))
#         mask = phi == p
#         print(theta[mask])
    wavelength = c / ff
    H_theta = wavelength * (50 / (np.pi * Z_0)) ** 0.5 * gain ** 0.5  # * np.exp(1j * phase)  ## do not use phases, this will screw up the interpolation
    H_phi = H_theta * 1e-3
    output_filename = '{}.pkl'.format(os.path.join(path, name, name))
    with open(output_filename, 'wb') as fout:
        logger.info('saving output to {}'.format(output_filename))
        pickle.dump([zen_boresight, azi_boresight, zen_ori, azi_ori, ff, theta, phi, H_phi, H_theta], fout, protocol=2)
# def parse_ROOT_file(f, orientation):
#     import ROOT
#     from array import array
#     import copy
#     nt = ROOT.TChain("AntTree")
#     nt.Add(f)
#
#     # find how many frequencies were simulated
#     N = array('i', [0])
#     nt.SetBranchAddress("N", N)
#     nt.GetEntry(0)
#
#     nfreq = copy.copy(N[0])
#     print(N)
#
#     thetas = array('f', [0])
#     phis = array('f', [0])
#     frequencies = array('f', nfreq * [0.])
#     gains = array('d', nfreq * [0.])
#     phases = array('d', nfreq * [0.])
#
#     nt.SetBranchAddress("thetas", thetas)
#     nt.SetBranchAddress("phis", phis)
#     nt.SetBranchAddress("gains", gains)
#     nt.SetBranchAddress("phases", phases)
#     nt.SetBranchAddress("frequencies", frequencies)
#
#     n = nt.GetEntries()
#     ntot = n * nfreq
#     ff2 = np.zeros(ntot)
#     tphis = np.zeros(ntot)
#     tthetas = np.zeros(ntot)
#     Ephis = np.zeros(ntot, dtype=np.complex)
#     Ethetas = np.zeros(ntot, dtype=np.complex)
#     for i in range(n):
#         nt.GetEntry(i)
#         ff2[i * nfreq: (i + 1) * nfreq] = frequencies
#         tphis[i * nfreq: (i + 1) * nfreq] = np.ones(nfreq) * np.deg2rad(phis)
#         tthetas[i * nfreq: (i + 1) * nfreq] = np.ones(nfreq) * np.deg2rad(thetas)
#         Ethetas[i * nfreq: (i + 1) * nfreq] = np.array(gains) * np.exp(-1j * np.array(phases))
#
#     print(ff2[0])
#     t2 = np.array([np.array(ff2), np.array(tphis), np.array(tthetas)])
#     tmp = np.array(t2.T,
#                    dtype=[('freq', '<f8'), ('phi', '<f8'), ('theta', '<f8')])
#     print(tmp)
#     index = np.argsort(tmp, order=('freq', 'phi', 'theta'))
#     print(index.shape)
#     print(index)
#     a = 1 / 0
#     print('bliubb')
#     for i in range(len(index)):
#         print(i, ff2[index[:, 0]][i], tphis[index[:, 1]][i], tthetas[index[:, 2]][i])
#     print(np.min(tphis), np.max(tphis))
#     print(np.unique(tphis))
#     print(np.unique(tthetas))
#     """ Second tree in file contains only Z as function of frequency,
#     it is the same for all phi and theta."""
#
#     ntt = ROOT.TChain("ZTree")
#     ntt.Add(f)
#     NN = array('i', [0])
#     Re_Z = array('d', nfreq * [0.])
#     ntt.SetBranchAddress("N", NN)
#     ntt.SetBranchAddress("Re_Z", Re_Z)
#     ntt.GetEntry(0)
#     Z = np.array(Re_Z)
#     ff = np.array(frequencies)
#
#     boresight, tines = np.loadtxt(orientation, delimiter=',')
#     zen_boresight, azi_boresight = hp.cartesian_to_spherical(*boresight)
#     zen_ori, azi_ori = hp.cartesian_to_spherical(*tines)
#
#     return zen_boresight, azi_boresight, zen_ori, azi_ori, ff, Z, np.array(ff2), np.deg2rad(np.array(tphis)), np.deg2rad(np.array(tthetas)), np.array(Ephis), np.array(Ethetas)
#
#
# def preprocess_ROOT(path):
#     from scipy import constants
#     from scipy.interpolate import interp1d
#     c = constants.c * units.m / units.s
#     Z_0 = 119.9169 * np.pi
#     split = os.path.split(os.path.dirname(path))
#     name = split[1]
#     path = split[0]
#     zen_boresight, azi_boresight, zen_ori, azi_ori, ff, Z, ff2, phi, theta, Ephi, Etheta = parse_ROOT_file(os.path.join(path, name, '{}.root'.format(name)),
#                                                                                                            os.path.join(path, name, '{}.orientation'.format(name)))
#
#     get_Z = interp1d(ff, Z, kind='nearest')
#     wavelength = c / ff2
#
#     H_phi = (2 * wavelength * get_Z(ff2) * Ephi) / (Z_0) / 1j
#     H_theta = (2 * wavelength * get_Z(ff2) * Etheta) / (Z_0) / 1j
#     print(H_theta)
#     output_filename = '{}.pkl2gzip'.format(os.path.join(path, name, name))
#     with gzip.open(output_filename, 'wb') as fout:
#         logger.info('saving output to {}'.format(output_filename))
#         pickle.dump([zen_boresight, azi_boresight, zen_ori, azi_ori, ff2, theta, phi, H_phi, H_theta], fout, protocol=2)


class AntennaPattern():

    def __init__(self, antenna_model, path=path_to_antennamodels):
        self.__name = antenna_model
        from time import time
        t = time()
        filename = os.path.join(path, antenna_model, "{}.pkl".format(antenna_model))
        self.__notfound = False
        try:
            self.zen_boresight, self.azi_boresight, self.zen_ori, self.azi_ori, ff, thetas, phis, H_phi, H_theta = get_WIPLD_antenna_response(filename)
        except IOError:
            self.__notfound = True
            logger.warning("antenna response for {} not found".format(antenna_model))
            return

        self.frequencies = np.unique(ff)
        self.frequency_lower_bound = self.frequencies[0]
        self.frequency_upper_bound = self.frequencies[-1]
#         print("{} frequencies from {} to {}".format(len(self.frequencies), self.frequency_lower_bound, self.frequency_upper_bound))

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
                    index = self.__get_index(iFreq, iTheta, iPhi)
#                     print(index, iFreq, iTheta, iPhi, np.rad2deg(phis[index]), np.rad2deg(thetas[index]))
                    if (phi != phis[index]):
                        print "phi angle has changed during theta loop"
                        print phi / units.deg, phis[index] / units.deg
                        raise Exception("phi angle has changed during theta loop")

                    if (theta != thetas[index]):
                        print "theta angle has changed during theta loop"
                        print theta / units.deg, thetas[index] / units.deg
                        raise Exception("theta angle has changed during theta loop")

                    if (freq != ff[index]):
                        print "theta angle has changed during theta loop"
                        print freq, ff[index]
                        raise

        logger.info('loading antenna file {} took {:.0f} seconds'.format(antenna_model, time() - t))

    def __get_index(self, iFreq, iTheta, iPhi):
        return iFreq * self.n_theta * self.n_phi + iPhi * self.n_theta + iTheta

    def __get_antenna_response_vectorized_raw(self, freq, theta, phi):
        """
        get vector effective length in WIPLD coordinate system
        """
#         print('initial phi', phi)
        while phi < self.phi_lower_bound:
            phi += 2 * np.pi
#             print('+2pi = ', phi)
        while phi > self.phi_upper_bound:
            phi -= 2 * np.pi
#             print('-2pi = ', phi)
#         phi[phi < self.phi_lower_bound] += 2 * np.pi
#         phi[phi > self.phi_upper_bound] -= 2 * np.pi
#         if((np.sum(freq < self.frequency_lower_bound) or np.sum(freq > self.frequency_upper_bound)) or
        if(hp.is_equal(theta, self.theta_upper_bound, rel_precision=1e-5)):
            theta = self.theta_upper_bound
        if(hp.is_equal(theta, self.theta_lower_bound, rel_precision=1e-5)):
            theta = self.theta_lower_bound
        if(((phi < self.phi_lower_bound) or (phi > self.phi_upper_bound)) or
           ((theta < self.theta_lower_bound) or (theta > self.theta_upper_bound))):
            print self.__name
            print "theta lower bound", self.theta_lower_bound, theta, self.theta_upper_bound
            print "phi lower bound", self.phi_lower_bound, phi, self.phi_upper_bound
#             print "freq lower bound", self.frequency_lower_bound, freq, self.frequency_upper_bound
            print "theta, phi or frequency out of range, returning (0,0j)"
            print freq, self.frequency_lower_bound, self.frequency_upper_bound
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
            self.VEL_theta[self.__get_index(iFrequency_lower, iTheta_lower, iPhi_lower)],
            self.VEL_theta[self.__get_index(iFrequency_lower, iTheta_lower, iPhi_upper)])
        VELp_freq_low_theta_low = interpolate_linear(
            phi, phi_lower, phi_upper,
            self.VEL_phi[self.__get_index(iFrequency_lower, iTheta_lower, iPhi_lower)],
            self.VEL_phi[self.__get_index(iFrequency_lower, iTheta_lower, iPhi_upper)])

        # theta up
        VELt_freq_low_theta_up = interpolate_linear(
            phi, phi_lower, phi_upper,
            self.VEL_theta[self.__get_index(iFrequency_lower, iTheta_upper, iPhi_lower)],
            self.VEL_theta[self.__get_index(iFrequency_lower, iTheta_upper, iPhi_upper)])
        VELp_freq_low_theta_up = interpolate_linear(
            phi, phi_lower, phi_upper,
            self.VEL_phi[self.__get_index(iFrequency_lower, iTheta_upper, iPhi_lower)],
            self.VEL_phi[self.__get_index(iFrequency_lower, iTheta_upper, iPhi_upper)])

        VELt_freq_low = interpolate_linear(theta, theta_lower,
                                                      theta_upper,
                                                      VELt_freq_low_theta_low,
                                                      VELt_freq_low_theta_up)
        VELp_freq_low = interpolate_linear(theta, theta_lower,
                                                      theta_upper,
                                                      VELp_freq_low_theta_low,
                                                      VELp_freq_low_theta_up)

        # upper frequency bound
        # theta low
        VELt_freq_up_theta_low = interpolate_linear(
            phi, phi_lower, phi_upper,
            self.VEL_theta[self.__get_index(iFrequency_upper, iTheta_lower, iPhi_lower)],
            self.VEL_theta[self.__get_index(iFrequency_upper, iTheta_lower, iPhi_upper)])
        VELp_freq_up_theta_low = interpolate_linear(
            phi, phi_lower, phi_upper,
            self.VEL_phi[self.__get_index(iFrequency_upper, iTheta_lower, iPhi_lower)],
            self.VEL_phi[self.__get_index(iFrequency_upper, iTheta_lower, iPhi_upper)])

        # theta up
        VELt_freq_up_theta_up = interpolate_linear(
            phi, phi_lower, phi_upper,
            self.VEL_theta[self.__get_index(iFrequency_upper, iTheta_upper, iPhi_lower)],
            self.VEL_theta[self.__get_index(iFrequency_upper, iTheta_upper, iPhi_upper)])
        VELp_freq_up_theta_up = interpolate_linear(
            phi, phi_lower, phi_upper,
            self.VEL_phi[self.__get_index(iFrequency_upper, iTheta_upper, iPhi_lower)],
            self.VEL_phi[self.__get_index(iFrequency_upper, iTheta_upper, iPhi_upper)])

        VELt_freq_up = interpolate_linear(theta, theta_lower, theta_upper,
                                                     VELt_freq_up_theta_low,
                                                     VELt_freq_up_theta_up)
        VELp_freq_up = interpolate_linear(theta, theta_lower, theta_upper,
                                                     VELp_freq_up_theta_low,
                                                     VELp_freq_up_theta_up)

        interpolated_VELt = interpolate_linear_vectorized(freq, frequency_lower,
                                                          frequency_upper,
                                                          VELt_freq_low,
                                                          VELt_freq_up)
        interpolated_VELp = interpolate_linear_vectorized(freq, frequency_lower,
                                                          frequency_upper,
                                                          VELp_freq_low,
                                                          VELp_freq_up)

        # set all out of bound frequencies to zero
        interpolated_VELt[out_of_bound_freqs_low] = 0 + 0 * 1j
        interpolated_VELt[out_of_bound_freqs_high] = 0 + 0 * 1j
        interpolated_VELp[out_of_bound_freqs_low] = 0 + 0 * 1j
        interpolated_VELp[out_of_bound_freqs_high] = 0 + 0 * 1j
        return interpolated_VELt, interpolated_VELp

    def __get_antenna_rotation(self, zen_boresight, azi_boresight, zen_ori, azi_ori):
        # define orientation of wiplD antenna simulation (in ARIANNA CS)
        e1 = hp.spherical_to_cartesian(self.zen_boresight, self.azi_boresight)  # boresight direction
        e2 = hp.spherical_to_cartesian(self.zen_ori, self.azi_ori)  # vector perpendicular to tine plane
        e3 = np.cross(e1, e2)
        E = np.array([e1, e2, e3])
#         logger.debug("antenna orientation in WIPLD= {}".format(E))
#         print('E', E)

        # get normal vectors for antenne orientation in field (in ARIANNA CS)
        a1 = hp.spherical_to_cartesian(zen_boresight, azi_boresight)
        a2 = hp.spherical_to_cartesian(zen_ori, azi_ori)
        a3 = np.cross(a1, a2)
        A = np.array([a1, a2, a3])
#         logger.debug("antenna orientation in field = {}".format(A))
        from numpy.linalg import inv
#         print('A', A, inv(A))
        return np.matmul(inv(E), A)

    def __get_theta_and_phi(self, zenith, azimuth, zen_boresight, azi_boresight, zen_ori, azi_ori):
        """
        transform zenith and azimuth angle in ARIANNA coordinate system to the WIPLD coordinate system.
        In addition the orientation of the antenna as deployed in the field is taken into account.
        """

        rot = self.__get_antenna_rotation(zen_boresight, azi_boresight, zen_ori, azi_ori)

        incoming_direction = hp.spherical_to_cartesian(zenith, azimuth)
        incoming_direction_WIPLD = np.dot(rot, incoming_direction.T).T
        theta, phi = hp.cartesian_to_spherical(*incoming_direction_WIPLD)
#         theta = 0.5 * np.pi - theta  # in wipl D the elevation is defined with 0deg being in the x-y plane
#         theta = hp.get_normalized_angle(theta)
#         phi = hp.get_normalized_angle(phi)

        logger.debug("zen/az {:.0f} {:.0f} transform to {:.0f} {:.0f}".format(zenith / units.deg,
                                                                              azimuth / units.deg,
                                                                              theta / units.deg,
                                                                              phi / units.deg))
        return theta, phi


    def get_antenna_response_dummy(self, freq, theta, phi, cutoff_freq = 50 * units.MHz):
        """
        Dummy LPDA model.
        Flat gain as function of frequency, no group delay. 
        Can be used instead of __get_antenna_response_vectorized_raw
        """       
        max_gain_co = 4 
        max_gain_cross = 2 #Check whether these values are actually reasonable
        
        index = np.argmax(freq > cutoff_freq)
        Gain = np.ones_like(freq)
        from scipy.signal import hann
        filter = hann(2*index)
        Gain[:index] = filter[:index]
        H_eff = 1./freq * Gain
    
        # at WIPL-D (1,0,0) Gain max for e_theta (?? I hope)
        # Standard units, deliver H_eff in meters
        Z_0 = constants.physical_constants['characteristic impedance of vacuum'][0] * units.ohm
        Z_ant = 50 * units.ohm
        
        #Assuming simple cosine, sine falls-off for dummy module
        H_eff_t = Gain * max_gain_cross * 1/freq
        H_eff_t *= np.cos(theta) * np.cos(phi) 
        H_eff_t *= constants.c * units.m / units.s * Z_ant / Z_0 / np.pi
        
        H_eff_p = Gain * max_gain_co * 1/freq
        H_eff_p *= np.cos(theta) * np.cos(phi) 
        H_eff_p *= constants.c * units.m / units.s * Z_ant / Z_0 / np.pi
    

#         import matplotlib.pyplot as plt
#         print theta, phi
#         plt.plot(H_eff_t)
#         plt.plot(H_eff_p)
#         plt.show()
#         1/0       
        
        
        return H_eff_p, H_eff_t

    def get_antenna_response_vectorized(self, freq, zenith, azimuth, zen_boresight, azi_boresight, zen_ori, azi_ori):
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
        zen_boresight : float
            zenith angle of the boresight direction of the antenna. Specifies the orientation of the antenna in the field
        azi_boresight : float
            azimuth angle of the boresight direction of the antenna. Specifies the orientation of the antenna in the field
        zen_ori : float
            zenith angle of the vector perpendicular to the plane defined by the antenna tines, and into the direction of the connector
        azi_ori : float
            azimuth angle of the vector perpendicular to the plane defined by the antenna tines, and into the direction of the connector

        Returns
        -------
        VEL: dictonary of complex arrays
            theta and phi component of the vector effective length, both components
            are complex floats or arrays of complex floats
            of the same length as the frequency input
        """
        if self.__notfound:
            VEL = {}
            VEL['theta'] = np.ones(len(freq), dtype=np.complex)
            VEL['phi'] = np.ones(len(freq), dtype=np.complex)
            return VEL

        if(isinstance(freq, (float, int))):
            freq = np.array([freq])
        theta, phi = self.__get_theta_and_phi(zenith, azimuth, zen_boresight, azi_boresight, zen_ori, azi_ori)
#         print('get_antenna_response_vectorized', zenith, azimuth, theta, phi)
        Vtheta_raw, Vphi_raw = self.__get_antenna_response_vectorized_raw(freq, theta, phi)

        # now rotate the raw theta and phi component of the VEL into the ARIANNA coordinate system.
        # As the theta and phi angles are differently defined in WIPLD and ARIANNA, also the orientation of the
        # eTheta and ePhi unit vectors are different.
        cstrans = cs.cstrafo(zenith=theta, azimuth=phi)
        V_xyz_raw = cstrans.transform_from_onsky_to_ground(np.array([np.zeros(Vtheta_raw.shape[0]), Vtheta_raw, Vphi_raw]))
        rot = self.__get_antenna_rotation(zen_boresight, azi_boresight, zen_ori, azi_ori)
        from numpy.linalg import inv
        V_xyz = np.dot(inv(rot), V_xyz_raw)
#         V_xyz = V_xyz_raw

        cstrans2 = cs.cstrafo(zenith=zenith, azimuth=azimuth)
        V_onsky = cstrans2.transform_from_ground_to_onsky(V_xyz)
        VEL = {}
        VEL['theta'] = V_onsky[1]
        VEL['phi'] = V_onsky[2]
        return VEL


class AntennaPatternProvider(object):
    __instance = None

    def __new__(cls):
        if AntennaPatternProvider.__instance is None:
            AntennaPatternProvider.__instance = object.__new__(cls)
        return AntennaPatternProvider.__instance

    def __init__(self):
        self.__open_antenna_patterns = {}
        self.__antenna_model_replacements = {}

        antenna_directory = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(antenna_directory, 'antenna_model_replacements.json')
        if(os.path.exists(filename)):
            with open(filename, 'r') as fin:
                self.__antenna_model_replacements = json.load(fin)

    def load_antenna_pattern(self, name):
        if(name in self.__antenna_model_replacements.keys()):
            name = self.__antenna_model_replacements[name]
        if (name not in self.__open_antenna_patterns.keys()):
            self.__open_antenna_patterns[name] = AntennaPattern(name)
        return self.__open_antenna_patterns[name]

# class AntennaPatternProvider:
#
#     class __AntennaPatternProvider:
#
#         def __init__(self):
#             self.val = arg
#
#         def __str__(self):
#             return repr(self) + self.val
#
#     instance = None
#
#     def __init__(self, arg):
#         if not AntennaPatternProvider.instance:
#             AntennaPatternProvider.instance = AntennaPatternProvider.__AntennaPatternProvider(arg)
#         else:
#             AntennaPatternProvider.instance.val = arg
#
#     def __getattr__(self, name):
#         return getattr(self.instance, name)
