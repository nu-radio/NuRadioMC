import os
import itertools
import functools
import numpy as np
from scipy import constants
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid

from NuRadioReco.utilities import units
from NuRadioReco.utilities import dataservers

import logging
logger = logging.getLogger("NuRadioMC.cross_sections")


@functools.lru_cache(maxsize=1)
def _read_differential_cross_section_BGR18():
    """
    Read the differential cross section dsigma / dy.
    """

    # shape of dsigma_dy_ref (flavor, cc_nc, energy, inelaticity)
    filepath = os.path.join(os.path.dirname(__file__), "data", 'BGR18_dsigma_dy_H2O.npz')
    if not os.path.exists(filepath):
        logger.status("Downloading BGR18 differential cross sections (first-time setup)...")
        dataservers.download_from_dataserver('cross_sections/BGR18_dsigma_dy_H2O.npz', filepath, unpack_tarball=False)
    data = np.load(filepath)

    # Convert to NuRadio units. We have to divide by 18 because the differential cross section
    # is given for the interaction between neutrinos and ice nuclei (which carry 18 nucleons)
    # while in NuRadio we use the cross section per nucleon.
    dsigma_dy_ref = data['dsigma_dy_ref'] * units.cm2 / 18

    flavors_ref = data['flavors_ref']
    nu_energies_ref = data['nu_energies_ref']
    yy_ref = data['y_ref']
    ncccs_ref = data['ncccs_ref']

    return nu_energies_ref, yy_ref, flavors_ref, ncccs_ref, dsigma_dy_ref

@functools.lru_cache(maxsize=2)
def _integrate_over_differential_cross_section_BGR18(simple=False):
    """
    Integrate the differential cross section dsigma / dy over y.
    """

    # shape of dsigma_dy_ref (flavor, cc_nc, energy, inelasticity)
    nu_energies_ref, yy_ref, flavors_ref, ncccs_ref, dsigma_dy_ref = _read_differential_cross_section_BGR18()

    if simple:
        dsigma_dy_integrated = trapezoid(dsigma_dy_ref, yy_ref, axis=-1)
    else:
        dsigma_dy_integrated = integrate_pwpl(dsigma_dy_ref, yy_ref, low=0, high=1)

    # Extend the cross section to include the total cross section by summing nc and cc contributions
    cross_section = np.zeros((len(flavors_ref), 3, len(nu_energies_ref)))
    cross_section[:, :2, :] = dsigma_dy_integrated
    cross_section[:, 2, :] = dsigma_dy_integrated[:, 0, :] + dsigma_dy_integrated[:, 1, :]
    ncccs_ref = np.append([ele.lower() for ele in ncccs_ref], 'total')

    return nu_energies_ref, flavors_ref, ncccs_ref, cross_section


def param(energy, inttype='cc', parameterization='ctw'):
    """
    Parameterization and constants as used in get_nu_cross_section()
    See documentation there for details.

    """
    if np.any(energy < 1e4 * units.GeV):
        logger.warning(
            "CTW / BGR neutrino nucleon cross sections not valid for energies below 1e4 GeV, "
            f"({energy / units.GeV}GeV was requested)")

        if hasattr(energy, "__len__"):
            return np.nan * np.ones_like(energy)
        else:
            return np.nan

    if parameterization == 'ctw':
        """
        Phys.Rev.D83:113009,2011 Amy Connolly, Robert S. Thorne, David Waters
        """
        if inttype == 'cc':
            c = (-1.826, -17.31, -6.406, 1.431, -17.91)  # nu, CC
        elif inttype == 'nc':
            c = (-1.826, -17.31, -6.448, 1.431, -18.61)  # nu, NC
        elif inttype == 'cc_bar':
            c = (-1.033, -15.95, -7.247, 1.569, -17.72)  # nu_bar, CC
        elif inttype == 'nc_bar':
            c = (-1.033, -15.95, -7.296, 1.569, -18.30)  # nu_bar, NC

        elif inttype == 'nc_up':
            c = (-1.456, 32.23, -32.32, 5.881, -49.41)  # nu, NC
        elif inttype == 'cc_up':
            c = (-1.456, 33.47, -33.02, 6.026, -49.41)  # nu, CC
        elif inttype == 'nc_bar_up':
            c = (-2.945, 143.2, -76.70, 11.75, -142.8)  # nu_bar, NC
        elif inttype == 'cc_bar_up':
            c = (-2.945, 144.5, -77.44, 11.9, -142.8)  # nu_bar, CC
        elif inttype == 'nc_down':
            c = (-15.35, 16.16, 37.71, -8.801, -253.1)  # nu, NC
        elif inttype == 'cc_down':
            c = (-15.35, 13.86, 39.84, -9.205, -253.1)  # nu, CC
        elif inttype == 'nc_bar_down':
            c = (-13.08, 15.17, 31.19, -7.757, -216.1)  # nu_bar, NC
        elif inttype == 'cc_bar_down':
            c = (-13.08, 12.48, 33.52, -8.191, -216.1)  # nu_bar, CC
        else:
            logger.error("Type {0} of interaction not defined for 'ctw'".format(inttype))
            raise NotImplementedError
    else:
        logger.error("Parameterization {0} of interaction cross section not defined".format(parameterization))
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
    if isinstance(inttype, str):
        inttype = np.array([inttype] * energy.shape[0])

    if isinstance(flavors, (int, np.integer)):
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

    crscn[particles_cc] = neutrino_cc(energy[particles_cc])
    crscn[particles_nc] = neutrino_nc(energy[particles_nc])
    crscn[antiparticles_cc] = antineutrino_cc(energy[antiparticles_cc])
    crscn[antiparticles_nc] = antineutrino_nc(energy[antiparticles_nc])

    return crscn


def get_nu_cross_section(energy, flavors, inttype='total', cross_section_type='hedis_bgr18'):
    """ Returns neutrino cross-section

    Parameters
    ----------
    energy: float / array of floats
        neutrino energies/momenta in standard units

    flavors: float / array of floats
        neutrino flavor (integer) encoded as using PDG numbering scheme,
        particles have positive sign, anti-particles have negative sign, relevant are:

        * 12: electron neutrino
        * 14: muon neutrino
        * 16: tau neutrino

    inttype: str, array of str
        interaction type. Options:

        * nc : neutral current
        * cc : charged current
        * total: total (for non-array type)
        * total_up : (only for ctw) total cross-section up uncertainty
        * total_down : (only for ctw) total cross-section down uncertainty

    cross_section_type: {'ctw', 'ghandi', 'csms', 'hedis_bgr18'}, default 'hedis_bgr18'
        defines model of cross-section. Options:

        * ctw : A. Connolly, R. S. Thorne, and D. Waters, Phys. Rev.D 83, 113009 (2011).
          cross-sections for all interaction types and flavors
        * ghandi : according to Ghandi et al. Phys.Rev.D58:093009,1998
          only one cross-section for all interactions and flavors
        * csms : A. Cooper-Sarkar, P. Mertsch, S. Sarkar, JHEP 08 (2011) 042
        * hedis_bgr18 : Parameterization from arXiv:2004.04756v2 (prepared for JCAP)

    Returns
    -------
    crscn: float / array of floats
        Cross-section in m^2
    """

    if cross_section_type == 'ghandi':
        crscn = 7.84e-36 * units.cm ** 2 * np.power(energy / units.GeV, 0.363)

    elif cross_section_type == 'hedis_bgr18':
        nu_energies_ref, flavors_ref, ncccs_ref, cross_section_ref = _integrate_over_differential_cross_section_BGR18(simple=False)

        if np.any(energy > nu_energies_ref[-1]):
            raise ValueError(
                f"Exceeding energy limit of BGR18 cross-section parameterization (E_lim = {nu_energies_ref[-1]:.2e} eV). "
                "Please use a different cross-section model.")

        crscn = np.zeros_like(energy)
        for flav, it in itertools.product(np.unique(flavors), np.unique(inttype)):
            # If flavors and inttype are not arrays, you want mask to be all True
            mask = np.ones_like(energy, dtype=bool)
            if isinstance(flavors, np.ndarray):
                mask = np.logical_and(mask, flavors == flav)

            if isinstance(inttype, np.ndarray):
                mask = np.logical_and(mask, inttype == it)

            idx_flav = int(np.squeeze(np.argwhere(flavors_ref == flav)))
            idx_inttype = int(np.squeeze(np.argwhere(ncccs_ref == it.lower())))

            if isinstance(energy, np.ndarray):
                crscn[mask] = 10 ** interp1d(
                    nu_energies_ref, np.log10(cross_section_ref[idx_flav, idx_inttype]),
                    bounds_error=True)(energy[mask])
            else:
                crscn[mask] = 10 ** interp1d(
                    nu_energies_ref, np.log10(cross_section_ref[idx_flav, idx_inttype]),
                    bounds_error=True)(energy)

    elif cross_section_type == 'ctw':
        crscn = np.zeros_like(energy)
        if isinstance(inttype, str):
            if inttype == 'total':

                if isinstance(flavors, (int, np.integer)):
                    if flavors >= 0:
                        crscn = param(energy, 'nc', parameterization=cross_section_type) + param(energy, 'cc', parameterization=cross_section_type)
                    else:
                        crscn = param(energy, 'nc_bar', parameterization=cross_section_type) + param(energy, 'cc_bar', parameterization=cross_section_type)
                else:
                    antiparticles = np.where(flavors < 0)
                    particles = np.where(flavors >= 0)

                    crscn[particles] = param(energy[particles], 'nc', parameterization=cross_section_type) + param(energy[particles], 'cc', parameterization=cross_section_type)
                    crscn[antiparticles] = param(energy[antiparticles], 'nc_bar', parameterization=cross_section_type) + param(energy[antiparticles], 'cc_bar', parameterization=cross_section_type)

            elif inttype == 'total_up':

                if isinstance(flavors, (int, np.integer)):
                    if flavors >= 0:
                        crscn = param(energy, 'nc_up') + param(energy, 'cc_up', parameterization=cross_section_type)
                    else:
                        crscn = param(energy, 'nc_bar_up') + param(energy, 'cc_bar_up', parameterization=cross_section_type)
                else:
                    antiparticles = np.where(flavors < 0)
                    particles = np.where(flavors >= 0)

                    crscn[particles] = param(energy[particles], 'nc_up') + param(energy[particles], 'cc_up', parameterization=cross_section_type)
                    crscn[antiparticles] = param(energy[antiparticles], 'nc_bar_up') + param(energy[antiparticles], 'cc_bar_up', parameterization=cross_section_type)

            elif inttype == 'total_down':

                if isinstance(flavors, (int, np.integer)):
                    if flavors >= 0:
                        crscn = param(energy, 'nc_down') + param(energy, 'cc_down', parameterization=cross_section_type)
                    else:
                        crscn = param(energy, 'nc_bar_down') + param(energy, 'cc_bar_down', parameterization=cross_section_type)
                else:
                    antiparticles = np.where(flavors < 0)
                    particles = np.where(flavors >= 0)

                    crscn[particles] = param(energy[particles], 'nc_down') + param(energy[particles], 'cc_down', parameterization=cross_section_type)
                    crscn[antiparticles] = param(energy[antiparticles], 'nc_bar_down') + param(energy[antiparticles], 'cc_bar_down', parameterization=cross_section_type)

            else:
                if isinstance(flavors, (int, np.integer)):
                    crscn = param(energy, inttype, parameterization=cross_section_type)
                else:
                    antiparticles = np.where(flavors < 0)
                    particles = np.where(flavors >= 0)
                    crscn[particles] = param(energy[particles], inttype, parameterization=cross_section_type)
                    crscn[antiparticles] = param(energy[antiparticles], inttype, parameterization=cross_section_type)
        else:

                if isinstance(flavors, (int, np.integer)):

                    particles_cc = np.where(inttype == 'cc')
                    particles_nc = np.where(inttype == 'nc')
                    if flavors >= 0:
                        crscn[particles_cc] = param(energy[particles_cc], 'cc', parameterization=cross_section_type)
                        crscn[particles_nc] = param(energy[particles_nc], 'nc', parameterization=cross_section_type)
                    else:
                        crscn[particles_cc] = param(energy[particles_cc], 'cc_bar', parameterization=cross_section_type)
                        crscn[particles_nc] = param(energy[particles_nc], 'nc_bar', parameterization=cross_section_type)

                else:
                    particles_cc = np.where((flavors >= 0) & (inttype == 'cc'))
                    particles_nc = np.where((flavors >= 0) & (inttype == 'nc'))
                    antiparticles_cc = np.where((flavors < 0) & (inttype == 'cc'))
                    antiparticles_nc = np.where((flavors < 0) & (inttype == 'nc'))

                    crscn[particles_cc] = param(energy[particles_cc], 'cc', parameterization=cross_section_type)
                    crscn[particles_nc] = param(energy[particles_nc], 'nc', parameterization=cross_section_type)
                    crscn[antiparticles_cc] = param(energy[antiparticles_cc], 'cc_bar', parameterization=cross_section_type)
                    crscn[antiparticles_nc] = param(energy[antiparticles_nc], 'nc_bar', parameterization=cross_section_type)

    elif cross_section_type == 'csms':
        crscn = csms(energy, inttype, flavors)

    else:
        logger.error("Cross-section {} not defined".format(cross_section_type))
        raise NotImplementedError

    return crscn


def get_interaction_length(
        Enu, density=.917 * units.g / units.cm ** 3, flavor=12, inttype='total',
        cross_section_type='hedis_bgr18'):
    """
    calculates interaction length from cross section

    Parameters
    ----------
    Enu: float
        neutrino energy
    density: float (optional)
        density of the medium, default density of ice = 0.917 g/cm**3
    flavors: float / array of floats
        Neutrino flavor (integer) encoded as using PDG numbering scheme.
        For more information see get_nu_cross_section()

    inttype: str, array of str
        interaction type.  For options see get_nu_cross_section()

    cross_section_type: str (default: 'hedis_bgr18')
        Defines model of cross-section. For options see get_nu_cross_section()

    Returns
    -------
    L_int: float
        interaction length
    """
    m_n = constants.m_p * units.kg  # nucleon mass, assuming proton mass
    L_int = m_n / get_nu_cross_section(Enu, flavors=flavor, inttype=inttype, cross_section_type=cross_section_type) / density
    return L_int

def integrate_pwpl(y, x, low=None, high=None, full_output=False):
    """
    Integrate y over x, assuming y(x) is a piecewise-continuous power law.

    Analytic integral of y(x)dx, assuming y(x) is a piecewise-continuous power law;
    that is, ``y(x) = A x**b`` with different values for ``A`` and ``b`` between each
    pair of subsequent values.

    The integration is always over the last axis of ``y``, so ``x`` should either be of a compatible
    shape to ``y`` or be a 1D-array with length ``y.shape[-1]``.
    The integral limits are from ``x[0]`` to ``x[-1]`` unless ``low`` or ``high`` are given.

    Parameters
    ----------
    y : array of floats
        The values of the function to integrate.
    x : array of floats
        The values of the dependent variable to integrate over. Should match the
        last axis of ``y``, which is always the axis which is integrated over.
        Additionally, ``x`` should be sorted.
    low : float, optional
        Extend the lower limit of the integral from ``x[0]`` to ``low``,
        by linearly extrapolating (in log-log space).
        Note that ``low`` should satisfy ``0 <= low < x[0]``
    high : float, optional
        Extend the upper limit of the integral from ``x[-1]`` to ``high``,
        by linearly extrapolating (in log-log space).
        Note that ``high`` should satisfy  ``high > x[-1]``.
    full_output : bool, optional
        If True, returns additional output. Default is False.

    Returns
    -------
    res : float | array of floats
        The result of the integration over the last axis of ``y``.
    (integral, x) : tuple of arrays, optional
        (Only if ``full_output==True``)
        A tuple with the integral ``Y(x)`` and the (optionally extended
        to include ``low`` and ``high``) array ``x``. ``Y(x)`` is the
        value of the integral integrated from ``low`` to ``x``, and
        will have the same length as ``x`` in the last axis.

        The integral ``Y(x)`` may be useful if e.g. ``y(x)`` describes
        a probability distribution function (PDF), in which case ``Y(x)``
        is the cumulative distribution function (CDF).

    Notes
    -----
    The function ``y`` is assumed to be of the form
    :math:`y_i(x) = A_i x^{b_i}` for :math:`x_i <= x < x_{i+1}`;
    thus, the integrand :math:`Y_i = \int_{x_i}^{x_{i+1}} y(x) dx` on each interval is

    .. math:: Y_i = \\frac{A_i}{b_i+1} [x_{i+1}^{b_i+1} - x_i^{b_i+1}]

    """

    ## Calling np.log when y=0 results in a bunch of RuntimeWarnings.
    ## Therefore, we manually mask out these values and set them to np.nan,
    ## and just manually set the integrand to zero afterwards
    nanmask = y==0
    binmask = nanmask[...,1:] | nanmask[..., :-1] # we mask bins if either edge is 'invalid'

    logy = np.nan * np.zeros_like(y)
    logy[~nanmask] = np.log(y[~nanmask])
    logx = np.log(x)
    slope = np.diff(logy) / np.diff(logx)
    lognorm = logy[...,:-1] - slope * logx[...,:-1]

    integrand = np.exp(
        lognorm
        + np.log((x[1:]**(slope + 1) - x[:-1]**(slope + 1)) / (slope + 1))
    )

    integrand[binmask] = 0

    if low is not None:
        if low < 0:
            raise ValueError(f"Cannot use power-law integration for negative values.")
        elif (low == 0) and np.any(slope[...,0] <= -1):
            raise ValueError(f"Cannot integrate to x={low} because min(slope) {min(slope[...,0])} <= -1")
        int_low = np.exp(
            lognorm[...,0]
            + np.log(
                (x[0]**(slope[...,0] + 1) - low**(slope[...,0] + 1))
                / (slope[...,0] + 1))
        )
        # set integrand to 0 if the first or second value of y (used for the extrapolation) is zero
        int_low = np.where(binmask[...,0], 0, int_low)
        integrand = np.concatenate([np.asarray(int_low)[...,None], integrand], axis=-1)
        x = np.concatenate([np.atleast_1d(low), x], axis=-1)

    if high is not None:
        int_high = np.exp(
            lognorm[...,-1]
            + np.log(
                (high**(slope[...,-1] + 1) - x[-1]**(slope[...,-1] + 1))
                / (slope[...,-1] + 1))
        )
        # set integrand to 0 if the (second-to-)last value of y (used for the extrapolation) is zero
        int_high = np.where(binmask[...,-1], 0, int_high)
        integrand = np.concatenate([integrand, np.asarray(int_high)[...,None]], axis=-1)
        x = np.concatenate([x, np.atleast_1d(high)], axis=-1)

    res = np.sum(integrand, axis=-1)

    if full_output:
        integral = np.cumsum(integrand, axis=-1)

        return res, (np.insert(integral, 0, 0, axis=-1), x)

    return res


if __name__ == "__main__":  # this part of the code gets only executed it the script is directly called
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, height_ratios=[2.5, 1], figsize=(8, 5),
                            sharex=True, sharey="row", gridspec_kw={'hspace': 0.03, 'wspace': 0.03})

    energy = np.logspace(13, 19) * units.eV

    cc_ctw = get_nu_cross_section(energy, 14, inttype="cc", cross_section_type="ctw") / units.picobarn
    cc_csms = get_nu_cross_section(energy, 14, inttype="cc", cross_section_type='csms') / units.picobarn
    cc_hedis_bgr18 = get_nu_cross_section(energy, 14, inttype="cc", cross_section_type='hedis_bgr18') / units.picobarn

    axs[0, 0].loglog(energy / units.PeV, cc_ctw, lw=1, label='CTW')
    axs[0, 0].loglog(energy / units.PeV, cc_csms, lw=1, label='CSMS')
    axs[0, 0].loglog(energy / units.PeV, cc_hedis_bgr18, lw=1, label='HEDIS-BGR')

    axs[1, 0].plot(energy / units.PeV, cc_ctw / cc_ctw, color='C0', lw=1)
    axs[1, 0].plot(energy / units.PeV, cc_csms / cc_ctw, color='C1', lw=1)
    axs[1, 0].plot(energy / units.PeV, cc_hedis_bgr18 / cc_ctw, color='C2', lw=1)

    nc_ctw = get_nu_cross_section(energy, 14, inttype="nc", cross_section_type="ctw") / units.picobarn
    nc_csms = get_nu_cross_section(energy, 14, inttype="nc", cross_section_type='csms') / units.picobarn
    nc_hedis_bgr18 = get_nu_cross_section(energy, 14, inttype="nc", cross_section_type='hedis_bgr18') / units.picobarn

    axs[0, 1].loglog(energy / units.PeV, nc_ctw, lw=1, label='CTW')
    axs[0, 1].loglog(energy / units.PeV, nc_csms, lw=1, label='CSMS')
    axs[0, 1].loglog(energy / units.PeV, nc_hedis_bgr18, lw=1, label='HEDIS-BGR')

    axs[1, 1].plot(energy / units.PeV, nc_ctw / nc_ctw, color='C0', lw=1)
    axs[1, 1].plot(energy / units.PeV, nc_csms / nc_ctw, color='C1', lw=1)
    axs[1, 1].plot(energy / units.PeV, nc_hedis_bgr18 / nc_ctw, color='C2', lw=1)

    fig.supxlabel("Energy [PeV]")
    axs[0, 0].set_ylabel("cross-section [pb]")
    axs[1, 0].set_ylabel("residual")

    axs[0, 0].legend(title=r"$\sigma_{\nu N}(CC)$")
    axs[0, 1].legend(title=r"$\sigma_{\nu N}(NC)$")

    fig.align_ylabels(axs[:, 0])
    fig.tight_layout()

    for ax in axs.flatten():
        ax.grid()

    plt.show()
