import numpy as np
from NuRadioMC.utilities import units


def param(energy, cross_section_type='cc'):
    """
    Parameterization and constants as used in
    get_nu_cross_section()
    See documentation there for details

    """

    if cross_section_type == 'cc':
        c = (-1.826, -17.31, -6.406, 1.431, -17.91) # nu, CC
    elif cross_section_type == 'nc':
        c = (-1.826, -17.31, -6.448, 1.431, -18.61) # nu, NC
    elif cross_section_type == 'cc_bar':
        c = (-1.033, -15.95, -7.247, 1.569, -17.72) # nu_bar, CC
    elif cross_section_type == 'nc_bar':
        c = (-1.033, -15.95, -7.296, 1.569, -18.30) # nu_bar, NC
    else:
        logger.error("Type {0} of cross-section not defined".format(cross_section_type))
        raise NotImplementedError

    epsilon = np.log10(energy/units.GeV)
    l_eps = np.log(epsilon - c[0])
    crscn = c[1] + c[2] * l_eps + c[3] * l_eps**2 + c[4]/l_eps
    crscn = np.power(10,crscn) * units.cm**2
    return crscn


def get_nu_cross_section(energy, flavors, inttype='total', cross_section_type = 'ctw'):
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
    """


    if cross_section_type == 'ghandi':
        crscn = 7.84e-36 * units.cm**2 * np.power(energy/units.GeV,0.363)

    elif cross_section_type == 'ctw':
        crscn = np.zeros_like(energy)
        if inttype == 'total':

            if ( type(flavors) == int or type(flavors) == np.int64 ):
                if flavors >= 0:
                    crscn = param(energy,'nc') + param(energy,'cc')
                else:
                    crscn = param(energy,'nc_bar') + param(energy,'cc_bar')
            else:
                antiparticles = np.where(flavors < 0)
                particles = np.where(flavors >= 0)

                crscn[particles] = param(energy[particles],'nc') + param(energy[particles],'cc')
                crscn[antiparticles] = param(energy[antiparticles],'nc_bar') + param(energy[antiparticles],'cc_bar')

        else:
            if (inttype == 'cc') or (inttype =='nc'):
                if ( type(flavors) == int or type(flavors) == np.int64 ):
                    crscn = param(energy,inttype)
                else:
                    antiparticles = np.where(flavors < 0)
                    particles = np.where(flavors >= 0)
                    crscn[particles] = param(energy[particles],inttype)
                    crscn[antiparticles] = param(energy[antiparticles],inttype)
            else:
                if ( type(flavors) == int or type(flavors) == np.int64 ):

                    particles_cc = np.where(inttype == 'cc')
                    particles_nc = np.where(inttype == 'nc')
                    if flavors >= 0:
                        crscn[particles_cc] = param(energy[particles_cc],'cc')
                        crscn[particles_nc] = param(energy[particles_nc],'nc')
                    else:
                        crscn[particles_cc] = param(energy[particles_cc],'cc_bar')
                        crscn[particles_nc] = param(energy[particles_nc],'nc_bar')


                else:
                    particles_cc = np.where((flavors >= 0) & (inttype == 'cc'))
                    particles_nc = np.where((flavors >= 0) & (inttype == 'nc'))
                    antiparticles_cc = np.where((flavors < 0) & (inttype == 'cc'))
                    antiparticles_nc = np.where((flavors < 0) & (inttype == 'nc'))

                    crscn[particles_cc] = param(energy[particles_cc],'cc')
                    crscn[particles_nc] = param(energy[particles_nc],'nc')
                    crscn[antiparticles_cc] = param(energy[antiparticles_cc],'cc_bar')
                    crscn[antiparticles_nc] = param(energy[antiparticles_nc],'nc_bar')

    return crscn


if __name__=="__main__":  # this part of the code gets only executed it the script is directly called

    inttype = np.array(['nc','cc','nc','nc'])

    flavors = np.array([14,16,-14,15])

    energy = np.array([1e18,1e17,1e19,2e17])*units.eV

    print(get_nu_cross_section(energy, flavors))
