import numpy as np
from NuRadioReco.utilities import units
from scipy.interpolate import interp1d
from scipy import constants


def param(energy, inttype='cc'):
    """
    Parameterization and constants as used in
    get_nu_cross_section()
    See documentation there for details

    """

    if inttype == 'cc':
        c = (-1.826, -17.31, -6.406, 1.431, -17.91)  # nu, CC
    elif inttype == 'nc':
        c = (-1.826, -17.31, -6.448, 1.431, -18.61)  # nu, NC
    elif inttype == 'cc_bar':
        c = (-1.033, -15.95, -7.247, 1.569, -17.72)  # nu_bar, CC
    elif inttype == 'nc_bar':
        c = (-1.033, -15.95, -7.296, 1.569, -18.30)  # nu_bar, NC
    else:
        logger.error("Type {0} of interaction not defined".format(inttype))
        raise NotImplementedError

    epsilon = np.log10(energy / units.GeV)
    l_eps = np.log(epsilon - c[0])
    crscn = c[1] + c[2] * l_eps + c[3] * l_eps ** 2 + c[4] / l_eps
    crscn = np.power(10, crscn) * units.cm ** 2
    return crscn


def csms(energy, inttype, flavors):
    """
    Neutrino cross sections according to
    Amanda Cooper-Sarkar, Philipp Mertsch, Subir Sarkar
    JHEP 08 (2011) 042
    """
    if type(inttype) == str:
        inttype = np.array([inttype] * energy.shape[0])

    if (type(flavors) == int or type(flavors) == np.int64):
        flavors = np.array([flavors] * energy.shape[0])

    neutrino = np.array((
        [50, 0.32, 0.10],
        [100, 0.65, 0.20],
        [200, 1.3, 0.41],
        [500, 3.2, 1.0],
        [1000, 6.2, 2.0],
        [2000, 12., 3.8],
        [5000, 27., 8.6],
        [10000, 47., 15.],
        [20000, 77., 26.],
        [50000, 140., 49.],
        [100000, 210., 75.],
        [200000, 310., 110.],
        [500000, 490., 180.],
        [1e6, 690., 260.],
        [2e6, 950., 360.],
        [5e6, 1400., 540.],
        [1e7, 1900., 730.],
        [2e7, 2600., 980.],
        [5e7, 3700., 1400.],
        [1e8, 4800., 1900.],
        [2e8, 6200., 2400.],
        [5e8, 8700., 3400.],
        [1e9, 11000., 4400.],
        [2e9, 14000., 5600.],
        [5e9, 19000., 7600.],
        [1e10, 24000., 9600.],
        [2e10, 30000., 12000.],
        [5e10, 39000., 16000.],
        [1e11, 48000., 20000.],
        [2e11, 59000., 24000.],
        [5e11, 75000., 31000.]
        ))

    neutrino[:, 0] *= units.GeV
    neutrino[:, 1] *= units.picobarn  # CC
    neutrino[:, 2] *= units.picobarn  # NC

    neutrino_cc = interp1d(neutrino[:, 0], neutrino[:, 1], bounds_error=True)
    neutrino_nc = interp1d(neutrino[:, 0], neutrino[:, 2], bounds_error=True)

    antineutrino = np.array((
        [50, 0.15, 0.05],
        [100, 0.33, 0.12],
        [200, 0.69, 0.24],
        [500, 1.8, 0.61],
        [1000, 3.6, 1.20],
        [2000, 7., 2.4],
        [5000, 17., 5.8],
        [10000, 31., 11.],
        [20000, 55., 19.],
        [50000, 110., 39.],
        [100000, 180., 64.],
        [200000, 270., 99.],
        [500000, 460., 170.],
        [1e6, 660., 240.],
        [2e6, 920., 350.],
        [5e6, 1400., 530.],
        [1e7, 1900., 730.],
        [2e7, 2500., 980.],
        [5e7, 3700., 1400.],
        [1e8, 4800., 1900.],
        [2e8, 6200., 2400.],
        [5e8, 8700., 3400.],
        [1e9, 11000., 4400.],
        [2e9, 14000., 5600.],
        [5e9, 19000., 7600.],
        [1e10, 24000., 9600.],
        [2e10, 30000., 12000.],
        [5e10, 39000., 16000.],
        [1e11, 48000., 20000.],
        [2e11, 59000., 24000.],
        [5e11, 75000., 31000.]
    ))

    antineutrino[:, 0] *= units.GeV
    antineutrino[:, 1] *= units.picobarn  # CC
    antineutrino[:, 2] *= units.picobarn  # NC

    antineutrino_cc = interp1d(antineutrino[:, 0], antineutrino[:, 1], bounds_error=True)
    antineutrino_nc = interp1d(antineutrino[:, 0], antineutrino[:, 2], bounds_error=True)

    crscn = np.zeros_like(energy)

    particles_cc = np.where((flavors >= 0) & (inttype == 'cc'))
    particles_nc = np.where((flavors >= 0) & (inttype == 'nc'))
    antiparticles_cc = np.where((flavors < 0) & (inttype == 'cc'))
    antiparticles_nc = np.where((flavors < 0) & (inttype == 'nc'))
#
    crscn[particles_cc] = neutrino_cc(energy[particles_cc])
    crscn[particles_nc] = neutrino_nc(energy[particles_nc])
    crscn[antiparticles_cc] = antineutrino_cc(energy[antiparticles_cc])
    crscn[antiparticles_nc] = antineutrino_nc(energy[antiparticles_nc])

    return crscn


def get_nu_cross_section(energy, flavors, inttype='total', cross_section_type='ctw'):
    """
    return neutrino cross-section

    Parameters
        ----------
    energy: float/ array of floats
        neutrino energies/momenta in standard units

    flavors: float / array of floats
        neutrino flavor (integer) encoded as using PDG numbering scheme,
        particles have positive sign, anti-particles have negative sign, relevant are:
        12: electron neutrino
        14: muon neutrino
        16: tau neutrino

    inttype: str, array of str
        interaction type
        nc : neutral current
        cc : charged current
        total: total (for non-array type)

    cross_section_type: str
        defines model of cross-section
        ghandi : according to Ghandi et al. Phys.Rev.D58:093009,1998
                 only one cross-section for all interactions and flavors
        ctw    : A. Connolly, R. S. Thorne, and D. Waters, Phys. Rev.D 83, 113009 (2011).
                 cross-sections for all interaction types and flavors
        csms   : A. Cooper-Sarkar, P. Mertsch, S. Sarkar, JHEP 08 (2011) 042
    """

    if cross_section_type == 'ghandi':
        crscn = 7.84e-36 * units.cm ** 2 * np.power(energy / units.GeV, 0.363)

    elif cross_section_type == 'ctw':
        crscn = np.zeros_like(energy)
        if type(inttype) == str:
            if inttype == 'total':

                if (type(flavors) == int or type(flavors) == np.int64):
                    if flavors >= 0:
                        crscn = param(energy, 'nc') + param(energy, 'cc')
                    else:
                        crscn = param(energy, 'nc_bar') + param(energy, 'cc_bar')
                else:
                    antiparticles = np.where(flavors < 0)
                    particles = np.where(flavors >= 0)

                    crscn[particles] = param(energy[particles], 'nc') + param(energy[particles], 'cc')
                    crscn[antiparticles] = param(energy[antiparticles], 'nc_bar') + param(energy[antiparticles], 'cc_bar')

            if (inttype == 'cc') or (inttype == 'nc'):
                if (type(flavors) == int or type(flavors) == np.int64):
                    crscn = param(energy, inttype)
                else:
                    antiparticles = np.where(flavors < 0)
                    particles = np.where(flavors >= 0)
                    crscn[particles] = param(energy[particles], inttype)
                    crscn[antiparticles] = param(energy[antiparticles], inttype)
        else:

                if (type(flavors) == int or type(flavors) == np.int64):

                    particles_cc = np.where(inttype == 'cc')
                    particles_nc = np.where(inttype == 'nc')
                    if flavors >= 0:
                        crscn[particles_cc] = param(energy[particles_cc], 'cc')
                        crscn[particles_nc] = param(energy[particles_nc], 'nc')
                    else:
                        crscn[particles_cc] = param(energy[particles_cc], 'cc_bar')
                        crscn[particles_nc] = param(energy[particles_nc], 'nc_bar')

                else:
                    particles_cc = np.where((flavors >= 0) & (inttype == 'cc'))
                    particles_nc = np.where((flavors >= 0) & (inttype == 'nc'))
                    antiparticles_cc = np.where((flavors < 0) & (inttype == 'cc'))
                    antiparticles_nc = np.where((flavors < 0) & (inttype == 'nc'))

                    crscn[particles_cc] = param(energy[particles_cc], 'cc')
                    crscn[particles_nc] = param(energy[particles_nc], 'nc')
                    crscn[antiparticles_cc] = param(energy[antiparticles_cc], 'cc_bar')
                    crscn[antiparticles_nc] = param(energy[antiparticles_nc], 'nc_bar')

    elif cross_section_type == 'csms':
        crscn = csms(energy, inttype, flavors)

    else:
        logger.error("Cross-section {} not defined".format(cross_section_type))
        raise NotImplementedError

    return crscn


def get_interaction_length(Enu, density=.917 * units.g / units.cm ** 3, flavor=12, inttype='total',
                           cross_section_type='ctw'):
    """
    calculates interaction length from cross section

    Parameters
    ----------
    Enu: float
        neutrino energy
    density: float (optional)
        density of the medium, default density of ice = 0.917 g/cm**3
    flavors: float / array of floats
        neutrino flavor (integer) encoded as using PDG numbering scheme,
        particles have positive sign, anti-particles have negative sign, relevant are:
        12: electron neutrino
        14: muon neutrino
        16: tau neutrino

    inttype: str, array of str
        interaction type
        nc : neutral current
        cc : charged current
        total: total (for non-array type)

    cross_section_type: str
        defines model of cross-section
        ghandi : according to Ghandi et al. Phys.Rev.D58:093009,1998
                 only one cross-section for all interactions and flavors
        ctw    : A. Connolly, R. S. Thorne, and D. Waters, Phys. Rev.D 83, 113009 (2011).
                 cross-sections for all interaction types and flavors
        csms   : A. Cooper-Sarkar, P. Mertsch, S. Sarkar, JHEP 08 (2011) 042

    Returns float: interaction length

    """
    m_n = constants.m_p * units.kg  # nucleon mass, assuming proton mass
    L_int = m_n / get_nu_cross_section(Enu, flavors=flavor, inttype=inttype, cross_section_type=cross_section_type) / density
    return L_int


if __name__ == "__main__":  # this part of the code gets only executed it the script is directly called

    inttype = np.array(['cc'] * 10)

#     inttype = 'cc'

    flavors = np.array([14] * 10)

#     flavors = 14

    energy = np.array([1e12, 1e14, 1e16, 2e17, 6e17, 1e18, 5e18, 9e18, 1e19, 1e20]) * units.eV

    cc = get_nu_cross_section(energy, flavors, inttype=inttype) / units.picobarn

    cc_new = get_nu_cross_section(energy, flavors, inttype=inttype, cross_section_type='csms') / units.picobarn

    import matplotlib.pyplot as plt

    plt.figure()
    plt.loglog(energy / units.GeV, cc, label='CTW')
    plt.loglog(energy / units.GeV, cc_new, label='CSMS')
    plt.xlabel("Energy [GeV]")
    plt.ylabel("CC [pb]")
    plt.legend()

    plt.show()
