import numpy as np
import scipy.constants
from NuRadioReco.utilities import units
import logging
logger = logging.getLogger('fluxes')

def get_nu_cross_section(energy, cross_section_type = 'ctw'):
    """
    return neutrino cross-section

    Parameters
        ----------
    energy: neutrino energy

    cross_section_type: str
        defines model of cross-section
        ghandi : according to Ghandi et al. Phys.Rev.D58:093009,1998
        ctwcc  : A. Connolly, R. S. Thorne, and D. Waters, Phys. Rev.D 83, 113009 (2011).
                Charge Current Only
        ctwnc  : A. Connolly, R. S. Thorne, and D. Waters, Phys. Rev.D 83, 113009 (2011).
                Neutral Current Only
        ctw    : A. Connolly, R. S. Thorne, and D. Waters, Phys. Rev.D 83, 113009 (2011).
                Total Cross Section
    """

    def param(energy,cross_section_type = 'cc'):

        if cross_section_type == 'cc':
            c = (-1.826, -17.31, -6.406, 1.431, -17.91) # nu, CC
        elif cross_section_type == 'nc':
            c = (-1.826, -17.31, -6.448, 1.431, -18.61) # nu, NC
        else:
            logger.error("Type {0} of cross-section not defined".format(cross_section_type))
        epsilon = np.log10(energy/units.GeV)
        l_eps = np.log(epsilon - c[0])
        crscn = c[1] + c[2] * l_eps + c[3] * l_eps**2 + c[4]/l_eps
        crscn = np.power(10,crscn) * units.cm**2
        return crscn

    if cross_section_type == 'ghandi':
        crscn = 7.84e-36 * units.cm**2 * np.power(energy/units.GeV,0.363)
    elif cross_section_type == 'ctwcc':
        crscn = param(energy,'cc')
    elif cross_section_type == 'ctwnc':
        crscn = param(energy,'nc')
    elif cross_section_type == 'ctw':
        crscn = param(energy,'nc') + param(energy,'cc')
    else:
        logger.error('Type {0} of neutrino cross-section not defined'.format(cross_section_type))

    return crscn


def get_interaction_length(energy, cross_section_type = 'ctw'):
    """
    return interaction length at given energy, assuming a constant ice density
    """
    ice_density = 0.917 # units.g/units.cm**3 water equivalent is 1, so unitless
    N_A = scipy.constants.Avogadro * units.cm**-3
    L = 1./(N_A * ice_density * get_nu_cross_section(energy, cross_section_type=cross_section_type))
    return L

def get_limit_from_aeff(energy, aeff,
                        livetime,
                        signalEff = 1.00,
                        energyBinsPerDecade=1.000,
                        upperLimOnEvents=2.44):

    """
    Limit from effective volume

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

    """

    evtsPerFluxPerEnergy = aeff * signalEff
    evtsPerFluxPerEnergy *= livetime

    ul  = upperLimOnEvents / evtsPerFluxPerEnergy
    ul *= energyBinsPerDecade / np.log(10)
    ul /= energy

    return ul



def get_limit_flux(energy, veff,
                    livetime,
                    signalEff = 1.00,
                    energyBinsPerDecade=1.000,
                    upperLimOnEvents=2.44,
                    nuCrsScn='ctw'):

    """
    Limit from effective volume

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
    evtsPerFluxPerEnergy *= livetime
    evtsPerFluxPerEnergy /= get_interaction_length(energy, cross_section_type = nuCrsScn)

    ul  = upperLimOnEvents / evtsPerFluxPerEnergy
    ul *= energyBinsPerDecade / np.log(10)
    ul /= energy

    return ul

# def get_integrated_limit_flux(energy, veff,
#                               livetime,
#                               signalEff = 1.00,
#                               upperLimOnEvents=2.44,
#                               nuCrsScn='ctw'):
#
#     """
#     Limit from effective volume on E^2 flux plot
#
#     Parameters:
#         --------------
#     energy: array of floats
#         energies (the bin centers), the binning in log10(E) must be equidistant!
#     veff: array of floats
#         effective volumes
#     livetime: float
#         time used
#     signalEff: float
#         efficiency of signal reconstruction
#     energyBinsPerDecade: float
#         1 for decade bins, 2 for half-decade bins, etc.
#     upperLimOnEvents: float
#          2.3 for Neyman UL w/ 0 background,
#          2.44 for F-C UL w/ 0 background, etc
#     nuCrsScn: str
#         type of neutrino cross-section
#
#
#     """
#     energy = np.array(energy)
#     veff = np.array(veff)
#     logE = np.log10(energy)
#     dlogE = logE[1] - logE[0]
#
#     L = get_interaction_length(energy, cross_section_type = nuCrsScn)
# #     N = np.sum(veff/L/E) * np.log(10) * livetime * signalEff * dlogE
#     ul = upperLimOnEvents * dlogE/ (livetime * signalEff * np.log(10)) * np.sum(L/veff * energy) / len(energy)
#
#     return ul

def get_limit_e1_flux(energy, veff,
                    livetime,
                    signalEff = 1.00,
                    energyBinsPerDecade=1.000,
                    upperLimOnEvents=2.44,
                    nuCrsScn='ctw'):

    """
    Limit from effective volume on E^1 flux plot

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
    evtsPerFluxPerEnergy *= livetime
    evtsPerFluxPerEnergy /= get_interaction_length(energy, cross_section_type = nuCrsScn)

    ul  = upperLimOnEvents / evtsPerFluxPerEnergy
    ul *= energyBinsPerDecade / np.log(10)

    return ul

def get_limit_e2_flux(energy, veff,
                    livetime,
                    signalEff = 1.00,
                    energyBinsPerDecade=1.000,
                    upperLimOnEvents=2.44,
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
    return energy**2 * get_limit_flux(energy, veff, livetime, signalEff, energyBinsPerDecade, upperLimOnEvents,nuCrsScn)


def get_number_of_events_for_flux(energies, flux, Veff, livetime, nuCrsScn='ctw'):
    """
    calculates the number of expected neutrinos for a certain flux assumption

    Parameters
    -----------
    energies: array of floats
        energies (the bin centers), the binning in log10(E) must be equidistant!
    flux: array of floats
        the flux at energy logE
    Veff: array of floats
        the effective volume per energy logE
    livetime: float
        the livetime of the detector (including signal efficiency)

    Returns
    -------
    array of floats: number of events per energy bin
    """
    energies = np.array(energies)
    Veff = np.array(Veff)
    logE = np.log10(energies)
    dlogE = logE[1] - logE[0]
    return np.log(10) * livetime * flux * energies * Veff / get_interaction_length(energies, cross_section_type = nuCrsScn) * dlogE


def get_exposure(energy, Veff, field_of_view=2*np.pi):
    """
    calculate exposure from effective volume 
    
    Parameters
    ----------
    energy: float
        neutrino energy (needed to calculate interaction length)
    Veff: float
        effective volume
    field_of_view: float
        the field of view of the detector
    
    Returns
    float: exposure
    """
    return Veff / field_of_view / get_interaction_length(energy)

def get_integrated_exposure(exp_func, E_low, E_high):
    """
    calculates the integral int(E^-2 * exposure(E) dE)
    
    integration is performed in log space
    """
    
    def f(logE):
        E = 10 ** logE
        return exp_func(E) * np.log(E) / E
    
    from scipy import integrate
    i = integrate.quad(f, np.log10(E_low), np.log10(E_high))
    return i[0]

def get_fluence_limit(int_exp):
    """
    calculates the fluence limit for a integrated exposure
    """
    return 2.39/int_exp
    



if __name__=="__main__":  # this part of the code gets only executed it the script is directly called


    energy = 10**18 * units.eV
    veff = 2150 * units.km**3 * units.sr
    livetime = 5 *units.year

    print "Cross section: {} cm^2".format(get_nu_cross_section(energy, cross_section_type = 'ctw'))

    print "interaction length: {} km".format(get_interaction_length(energy, cross_section_type = 'ctw')/units.km)

    print "calculating flux limit for {time} years and Veff of {veff} km^3 sr".format(time=livetime/units.year,
                            veff = veff/ (units.km**3 * units.sr))
    print "Flux limit: {} GeV/(cm^2 s sr)".format(get_limit_e2_flux(energy,veff, livetime) / (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1))

    aeff = np.array([0.0032,0.033,0.43,3.1,21,68,167])* units.km**2 * units.sr
    energies = 10**np.array([18,18.5,19,19.5,20,20.5,21])* units.eV

    print "Flux limit: {} GeV/(cm^2 s sr)".format(energies**2*get_limit_from_aeff(energies,aeff,livetime) / (units.GeV * units.cm**-2 * units.second**-1 * units.sr**-1))







