import numpy as np
import scipy.constants
from NuRadioReco.utilities import units
import logging
logger = logging.getLogger('fluxes')

def get_nu_cross_section(energy, type = 'ctw'):
    """
    return neutrino cross-section

    Parameters
        ----------
    energy: neutrino energy

    type: str
        defines model of cross-section
        ghandi : according to Ghandi et al. Phys.Rev.D58:093009,1998
        ctwcc  : A. Connolly, R. S. Thorne, and D. Waters, Phys. Rev.D 83, 113009 (2011).
                Charge Current Only
        ctwnc  : A. Connolly, R. S. Thorne, and D. Waters, Phys. Rev.D 83, 113009 (2011).
                Neutral Current Only
        ctw    : A. Connolly, R. S. Thorne, and D. Waters, Phys. Rev.D 83, 113009 (2011).
                Total Cross Section
    """

    def param(energy,type = 'cc'):

        if type == 'cc':
            c = (-1.826, -17.31, -6.406, 1.431, -17.91) # nu, CC
        elif type == 'nc':
            c = (-1.826, -17.31, -6.448, 1.431, -18.61) # nu, NC
        else:
            logger.error("Type {0} of cross-section not defined".format(type))
        epsilon = np.log10(energy/units.GeV)
        l_eps = np.log(epsilon - c[0])
        crscn = c[1] + c[2] * l_eps + c[3] * l_eps**2 + c[4]/l_eps
        crscn = np.power(10,crscn) * units.cm**2
        return crscn

    if type == 'ghandi':
        crscn = 7.84e-36 * units.cm**2 * np.power(energy/units.GeV,0.363)
    elif type == 'ctwcc':
        crscn = param(energy,'cc')
    elif type == 'ctwnc':
        crscn = param(energy,'nc')
    elif type == 'ctw':
        crscn = param(energy,'nc') + param(energy,'cc')
    else:
        logger.error('Type {0} of neutrino cross-section not defined'.format(type))

    return crscn


def get_interaction_length(energy, type = 'ctw'):
    """
    return interaction length at given energy, assuming a constant ice density
    """
    ice_density = 0.917 # units.g/units.cm**3 water equivalent is 1, so unitless
    N_A = scipy.constants.Avogadro * units.cm**-3
    L = 1./(N_A * ice_density * get_nu_cross_section(energy, type=type))
    return L


def get_limit_e2_flux(energy, veff,
                    livetime,
                    signalEff = 1.00,
                    energyBinsPerDecade=1.000,
                    upperLimOnEvents=2.300,
                    nuCrsScn='ctw'):

    """
    Limit from effective volume on E^2 flux plot

    Parameters:
        --------------
    energy: array of floats
        neutrino energy
    veff: array of floats
        effective volumes
    livetime: float
        time used
    signalEff: float
        efficiency of signal reconstruction
    energyBinsPerDecade: float
        1 for decade bins, 2 for half-decade bins, etc.
    upperLimOnEvents: float
         2.3 for Neyman UL w/ 0 background,
         2.44 for F-C UL w/ 0 background, etc
    nuCrsScn: str
        type of neutrino cross-section


    """

    evtsPerFluxPerEnergy = veff * signalEff
    print "Veff", evtsPerFluxPerEnergy
    evtsPerFluxPerEnergy *= livetime
    evtsPerFluxPerEnergy /= get_interaction_length(energy, type = nuCrsScn)

    ul  = upperLimOnEvents / evtsPerFluxPerEnergy
    ul *= energyBinsPerDecade / np.log(10)
    ul *= energy

    return ul




debug = False

if debug:

    energy = 10**18 * units.eV
    veff = 2150 * units.km**3 * units.sr
    livetime = 5 *units.year

    print "Cross section", get_nu_cross_section(energy, type = 'ctw')

    print "interaction length", get_interaction_length(energy, type = 'ctw')/units.km

    print "calculating flux limit for {time} years and Veff of {veff} km^3 sr".format(time=livetime/units.year,
                            veff = veff/ (units.km**3 * units.sr))
    print "Flux limit: {} GeV/(cm^2 s sr)".format(get_limit_e2_flux(energy,veff, livetime) / (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1))


