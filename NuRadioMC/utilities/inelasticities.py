import functools
import numpy as np
from scipy import constants
from scipy import interpolate as intp

from NuRadioReco.utilities import units
from NuRadioMC.utilities import cross_sections

import logging
logger = logging.getLogger('NuRadioMC.inelasticities')

from NuRadioReco.utilities.constants import (
    e_mass, mu_mass, pi_mass, rho770_mass, a1_mass, rho1450_mass, tau_mass, G_F)


def get_neutrino_inelasticity(n_events, model="hedis_bgr18", rnd=None,
                              nu_energies=1 * units.EeV, flavors=12, ncccs="CC"):
    """
    Standard inelasticity for deep inelastic scattering used so far.
    Ported from ShelfMC

    Parameters
    ----------
    n_events: int
        Number of events to be returned
    model: string
        the inelasticity model to use
    rnd: random generator object
        if None is provided, a new default random generator object is initialized
    nu_energies: float or array (default: 1 EeV)
        Energy of the neutrino. If a float is provided, all events will have the same
        energy. If an array is provided, it must have the same length as n_events
    flavors: int or array (default: 12)
        The flavor of the neutrino. 12 = nu_e, 14 = nu_mu, 16 = nu_tau, -12 = anti-nu_e,
        -14 = anti-nu_mu, -16 = anti-nu_tau. Negative values are used for antineutrinos.
        If an array is provided, it must have the same length as n_events.
    ncccs: string or array (default: "CC")
        The interaction type: "CC" for charged current, "NC" for neutral current. If an
        array is provided, it must have the same length as n_events.

    Returns
    -------
    inelasticities: array
        Array with the inelasticities
    """
    rnd = rnd or np.random.default_rng()

    if model.lower() == "ctw":
        # based on shelfmc
        r1 = 0.36787944
        r2 = 0.63212056
        return (-np.log(r1 + rnd.uniform(0., 1., n_events) * r2)) ** 2.5

    elif model.lower() == "bgr18" or model.lower() == "hedis_bgr18":
        yy = np.zeros(n_events)

        nu_energies_ref, yy_ref, flavors_ref, ncccs_ref, dsigma_dy_ref = cross_sections._read_differential_cross_section_BGR18()

        if np.any(nu_energies > max(nu_energies_ref)):
            logger.warning(
                "You are requesting inelasticities for energies outside of the validity of the BGR18 model. "
                f"You requested maximum energy {max(np.atleast_1d(nu_energies)) / units.eV:.2g} eV. "
                f"Largest available energy is {max(nu_energies_ref)/units.eV:.2g}, returning result for {max(nu_energies_ref)/units.eV:.2g}.")

        energy_indicies = np.digitize(nu_energies, nu_energies_ref)
        energy_indicies = np.clip(energy_indicies, 0, len(nu_energies_ref) - 1)
        nu_energies_binned = nu_energies_ref[energy_indicies]
        uEE_binned = np.unique(nu_energies_binned)
        uFlavor = np.unique(flavors)
        uNCCC = np.unique(ncccs)

        for energy in uEE_binned:
            for flavor in uFlavor:
                for nccc in uNCCC:
                    mask = (energy == nu_energies_binned) & (flavor == flavors) & (nccc == ncccs)
                    size = n_events
                    if isinstance(mask, np.ndarray):
                        size = np.sum(mask)

                    nccc = nccc.upper()
                    iF = np.argwhere(flavors_ref == flavor)[0][0]
                    inccc = np.argwhere(ncccs_ref == nccc)[0][0]
                    iE = np.argmin(np.abs(energy - nu_energies_ref))

                    cdf, y = _get_inverse_cdf_interpolation(iF, inccc, iE)

                    randoms = rnd.uniform(0, 1, size=size)
                    # np.interp is the quickest interpolation, and linear interpolation differs from
                    # log interpolation by less than 1 per mille for the stored spacing in y
                    yy[mask] = np.interp(randoms, cdf, y)

        return yy

    else:
        raise AttributeError(f"inelasticity model {model} is not implemented.")


@functools.lru_cache(maxsize=int(2**(int(np.log2(200 * 6 * 2) + 1))))
def _get_inverse_cdf_interpolation(iF, inccc, iE):
    nu_energies_ref, yy_ref, flavors_ref, ncccs_ref, dsigma_dy_ref = cross_sections._read_differential_cross_section_BGR18()
    xsec_int, (xsec_int_full, y) = cross_sections.integrate_pwpl(
        dsigma_dy_ref[iF, inccc, iE], yy_ref, low=0, high=1, full_output=True)
    cdf = xsec_int_full / xsec_int

    return cdf, y


def get_ccnc(n_events, rnd=None, model="hedis_bgr18", energy=None, flavors=12):
    """
    Get the nature of the interaction current: cc or nc

    Parameters
    ----------
    n_events: int
        Number of events to be returned
    rnd: random generator object (default: None)
        If None is provided, a new default random generator object is initialized
    model: string (default: "hedis_bgr18")
        The cross section model to determine cc fraction. For options see cross_sections.py
    energy: float or array (default: None)
        Energy of the neutrino. If None is provided a constant value is used (only for CTW model).
    flavors: int (default: 12)
        The flavor of the neutrino. Only relevant for the BGR18 model. 12 = nu_e, 14 = nu_mu, 16 = nu_tau.
        Negative values are used for antineutrinos. For the CTW model, this parameter is ignored.

    Returns
    -------
    ccnc: array
        Array with 'cc' or 'nc'

    See Also
    --------
    NuRadioMC.utilities.cross_sections.get_nu_cross_section
    """
    rnd = rnd or np.random.default_rng()

    random_sequence = rnd.uniform(0., 1., n_events)

    if energy is None:
        assert model.lower() == "ctw", "Only CTW supports energy-independent cc/nc fraction Energy is required for BGR18 model"
        # Ported from Shelf MC
        # https://github.com/persic/ShelfMC/blob/daf56916d85de019e848f415c2e9f4643a744674/functions.cc#L1055-L1064
        # based on CTW cross sections https://link.aps.org/doi/10.1103/PhysRevD.83.113009
        cc_fraction = 0.7064
    else:
        if not isinstance(energy, (float, int)):
            assert len(energy) == n_events, "Energy must be a scalar or an array of the same length as n_events"

        # Flavor only relevant to determine if its a neutrino or antineutrino. For cross section ratio it
        # hopefully doesn't matter
        cc = cross_sections.get_nu_cross_section(energy, flavors=flavors, inttype="cc", cross_section_type=model.lower())
        nc = cross_sections.get_nu_cross_section(energy, flavors=flavors, inttype="nc", cross_section_type=model.lower())
        cc_fraction = cc / (cc + nc)

    ccnc = np.where(random_sequence <= cc_fraction, 'cc', 'nc')

    return ccnc


def random_tau_branch(rnd=None):
    """
    Calculates a random tau branch decay
    See http://dx.doi.org/10.1016/j.cpc.2013.04.001

    rnd: random generator object
        if None is provided, a new default random generator object is initialized

    Returns
    -------
    branch: string
        The corresponding decay branch
    """
    rnd = rnd or np.random.default_rng()

    branching_ratios = np.array([0.18, 0.18])
    branching = rnd.uniform(0, 1)
    if (branching < np.sum(branching_ratios[0:1])):
        # tau -> nu_tau + mu + nu_tau
        branch = 'tau_mu'
    elif (branching < np.sum(branching_ratios[0:2])):
        # tau -> nu_tau + e + nu_e
        branch = 'tau_e'
    else:
        # tau -> nu_tau + hadrons
        branch = 'tau_had'

    return branch


def inelasticity_tau_decay(tau_energy, branch, rnd=None):
    """
    Returns the hadronic or electromagnetic inelasticity for the tau decay
    See http://dx.doi.org/10.1016/j.cpc.2013.04.001
    and https://arxiv.org/pdf/1607.00193.pdf

    Parameters
    ----------
    tau_energy: float
        Tau energy at the moment of decay
    branch: string
        Type of tau decay: 'tau_mu', 'tau_e', 'tau_had'
    rnd: random generator object
        if None is provided, a new default random generator object is initialized

    Returns
    -------
    inelasticity: float
        The fraction of energy carried by the leptonic or hadronic products
    """
    rnd = rnd or np.random.default_rng()

    if (branch == 'tau_had'):

        branching = np.array([0.12, 0.26, 0.13, 0.13])
        rs = np.array([pi_mass, rho770_mass, a1_mass, rho1450_mass]) / tau_mass

        def g_pi(y, r):
            if (y < 0 or y > 1 - r ** 2):
                return 0.
            else:
                return -(2 * y - 1 + r) / (1 - r ** 2) ** 2

        def g_1(y, r):
            if (y < 0 or y > 1 - r ** 2):
                return 0.
            else:
                return -(2 * y - 1 + r) * (1 - 2 * r) / (1 - r) ** 2 / (1 + 2 * r)

        def g_0(y, r):
            if (y < 0 or y > 1 - r ** 2):
                return 0.
            else:
                return 1 / (1 - r)

        def y_distribution(y):
            pi_term = branching[0] * (g_pi(y, rs[0]) + g_0(y, rs[0]))
            # rest_terms = branching[1:]*(g_1(y,rs)+g_0(y,rs))
            rest_terms = [ branch * (g_1(y, r) + g_0(y, r)) for branch, r in zip(branching[1:], rs[1:]) ]
            return pi_term + np.sum(rest_terms)

        chosen_y = rejection_sampling(y_distribution, 0, 1, 3)

        return 1 - chosen_y

    elif (branch == 'tau_e' or branch == 'tau_mu'):

        mu = tau_mass
        if (branch == 'tau_e'):
            m_l = e_mass
        elif (branch == 'tau_mu'):
            m_l = mu_mass

        nu_max = (mu ** 2 + m_l ** 2) / 2 / mu

        # Fraction energy distibution in the decaying particle rest frame
        def x_distribution(x):
            if (x < m_l / nu_max or x > 1):
                return 0.
            else:
                factor = G_F ** 2 * mu ** 5 / 192 / np.pi ** 3
                return factor * (3 - 2 * x) * x ** 2

        chosen_x = rejection_sampling(x_distribution, 0, 1, x_distribution(1))
        chosen_cos = rnd.uniform(-1, 1)

        y_rest = chosen_x * nu_max / tau_mass
        # Transforming the rest inelasticity to the lab inelasticity
        y_lab = y_rest - np.sqrt(y_rest ** 2 - (m_l / mu) ** 2) * chosen_cos

        return y_lab


def rejection_sampling(f, xmin, xmax, ymax, rnd=None):
    """
    Draws a random number following a given distribution using
    a rejection sampling algorithm.

    Parameters
    ----------
    f: function
        Random distribution
    xmin: float
        Minimum value of the argument
    xmax: float
        Maximum value of the argument
    ymax: float
        Maximum function value to use for the rejection sample
        (e.g., the maximum of the function)
    rnd: random generator object
        if None is provided, a new default random generator object is initialized

    Returns
    -------
    x: float
        Random value from the distribution
    """
    rnd = rnd or np.random.default_rng()

    while True:
        x = rnd.uniform(xmin, xmax)
        y = rnd.uniform(0, ymax)
        if f(x) >= y:
            break

    return x


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from NuRadioMC.utilities.cross_sections import get_nu_cross_section

    energy = np.logspace(13, 19) * units.eV

    cc_ctw = get_nu_cross_section(energy, 14, inttype="cc", cross_section_type="ctw") / units.picobarn
    cc_csms = get_nu_cross_section(energy, 14, inttype="cc", cross_section_type='csms') / units.picobarn
    cc_hedis_bgr18 = get_nu_cross_section(energy, 14, inttype="cc", cross_section_type='hedis_bgr18') / units.picobarn

    nc_ctw = get_nu_cross_section(energy, 14, inttype="nc", cross_section_type="ctw") / units.picobarn
    nc_csms = get_nu_cross_section(energy, 14, inttype="nc", cross_section_type='csms') / units.picobarn
    nc_hedis_bgr18 = get_nu_cross_section(energy, 14, inttype="nc", cross_section_type='hedis_bgr18') / units.picobarn

    fig, ax = plt.subplots(1, 1)

    ax.plot(energy / units.PeV, cc_ctw / (cc_ctw + nc_ctw), color="C0", ls="-", label='CTW')
    ax.plot(energy / units.PeV, cc_csms / (cc_csms + nc_csms), color="C1", ls="-", label='CSMS')
    ax.plot(energy / units.PeV, cc_hedis_bgr18 / (cc_hedis_bgr18 + nc_hedis_bgr18), color="C2", ls="-", label='BGR18-HEDIS')

    energy2 = np.repeat(energy, 100000)
    for model in ["CTW", "CSMS", "hedis_bgr18"]:
        ccnc = get_ccnc(len(energy2), model=model, energy=energy2)
        ccnc = ccnc.reshape(len(energy), -1)
        cc_fraction = np.array([np.sum(ele == "cc") / len(ele) for ele in ccnc])
        ax.plot(energy / units.PeV, cc_fraction, lw=1, ls="--")

    ax.set_xlabel("Energy [PeV]")

    ax.axhline(0.7064, color="k", lw=1, ls="--", label="get_ccnc energy independent")

    ax.set_xscale("log")
    ax.set_ylabel("cc fraction")
    ax.legend()


    n_events = 3000000
    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, height_ratios=[2.5, 1], gridspec_kw={"hspace": 0.05})

    inelasticities = get_neutrino_inelasticity(n_events, model="CTW")

    bins = np.linspace(0, 1, 1001)
    n_ctw, bins, _ = ax.hist(inelasticities, bins=bins, histtype="step", weights=np.ones_like(inelasticities) / n_events, lw=1, label=f'CTW, mean: {np.mean(inelasticities):.3f}')
    x = bins[:-1] + np.diff(bins) / 2
    for idx, nu_energy in enumerate([0.1 * units.EeV, 1 * units.EeV, 10 * units.EeV]):
        inelasticities_cc_bgr = get_neutrino_inelasticity(n_events, model="BGR18", ncccs="CC", nu_energies=nu_energy)
        n, _ , _ = ax.hist(inelasticities_cc_bgr, bins=bins, histtype="step", ls="--", color=f"C{idx+1}",
                weights=np.ones_like(inelasticities_cc_bgr) / n_events, lw=1, label=f'BGR18 CC {nu_energy / units.EeV:.2f}EeV, mean: {np.mean(inelasticities_cc_bgr):.3f}')

        ax2.plot(x, n / n_ctw, color=f"C{idx+1}", lw=1, ls="--")


        inelasticities_nc_bgr = get_neutrino_inelasticity(n_events, model="BGR18", ncccs="NC", nu_energies=nu_energy)
        n, _ , _ = ax.hist(inelasticities_nc_bgr, bins=bins, histtype="step", ls=":", color=f"C{idx+1}",
                weights=np.ones_like(inelasticities_nc_bgr) / n_events, lw=1, label=f'BGR18 NC {nu_energy / units.EeV:.2f}EeV, mean: {np.mean(inelasticities_nc_bgr):.3f}')
        ax2.plot(x, n / n_ctw, color=f"C{idx+1}", lw=1, ls=":")


    # plotting.draw_residual(ax, ax2)
    ax.grid()
    ax2.set_xlabel('Inelasticity')
    ax2.grid()
    ax2.set_ylabel("Ratio to CTW")
    ax.set_ylabel("probability density")

    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.align_ylabels([ax, ax2])


    fig2, (ax, ax2) = plt.subplots(2, 1, sharex=True, height_ratios=[2.5, 1], gridspec_kw={"hspace": 0.05})

    n_events = 1000000
    energies = np.logspace(16, 19, 100) * units.eV

    ineal_mean_ctw = np.ones_like(energies) * np.mean(get_neutrino_inelasticity(n_events, model="CTW"))
    ax.plot(energies / units.PeV, ineal_mean_ctw, color="k", lw=1, label="CTW")

    ineal_mean_cc = []
    ineal_mean_nc = []
    for e in energies:
        ineal_mean_cc.append(np.mean(get_neutrino_inelasticity(n_events, model="BGR18-HEDIS", ncccs="CC", nu_energies=e)))
        ineal_mean_nc.append(np.mean(get_neutrino_inelasticity(n_events, model="BGR18-HEDIS", ncccs="nc", nu_energies=e)))

    ax.plot(energies / units.PeV, ineal_mean_cc, label="BGR18-HEDIS CC", lw=1)
    ax.plot(energies / units.PeV, ineal_mean_nc, label="BGR18-HEDIS NC", lw=1)


    ax2.plot(energies / units.PeV, np.array(ineal_mean_cc) / ineal_mean_ctw, label="BGR18-HEDIS CC", lw=1)
    ax2.plot(energies / units.PeV, np.array(ineal_mean_nc) / ineal_mean_ctw, label="BGR18-HEDIS NC", lw=1)

    ax.set_xscale("log")
    ax.legend()
    ax.grid()

    ax.set_ylabel(r"$\langle y \rangle$")
    ax2.set_xlabel(r"$E_\nu$ [PeV]")
    ax2.set_ylabel(r"$\langle y \rangle$ ratio to CTW")
    ax2.grid()

    fig.tight_layout()
    fig.align_ylabels([ax, ax2])

    plt.show()
