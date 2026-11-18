"""
Limit plot: published limits on the diffuse neutrino flux, sensitivities of planned experiments and neutrino flux
models (from :mod:`NuRadioMC.utilities.neutrino_fluxes`) in E^2 dN/dE, to which you add the sensitivity of your
detector.

Use it from Python::

    from NuRadioMC.examples.Sensitivities import limit_plot
    fig, ax = limit_plot.get_limit_figure(diffuse=True)
    labels = limit_plot.add_limit(ax, [], energies, veff_sr, n_stations=100, label="my detector", livetime=5 * units.year)
    ax.legend(handles=labels, loc=2)

or from the command line, with an effective volume file: ``python limit_plot.py --help``.

Normalisation: limits and sensitivities depend on the width of the energy bins they are quoted for. All of them are
shown for the same binning, ``energy_bins_per_decade`` bins per decade (default 1, i.e. one decade wide bins; it
also applies to :func:`add_limit`). Each file in ``experiments`` states its own binning in the header line
"# Energy bins per decade: <n>", and its values are multiplied by ``energy_bins_per_decade / n``. Fluxes
(models, measurements) do not depend on the binning.
"""
import argparse
import functools
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from NuRadioReco.utilities import cr_flux, units
from NuRadioMC.utilities import fluxes, neutrino_fluxes

local_dir = os.path.abspath(os.path.dirname(__file__))

legendfontsize = 11

plotUnitsEnergy = units.eV
plotUnitsEnergyStr = "eV"
plotUnitsFlux = units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1
_E2_unit = units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1  # E^2 dN/dE unit of most tables

'''
Regarding ANITA Limits
# ====================================================

ANITA uses a *super* unusual differential limit bin width.

A limit is generally given by:

     E dN                           Sup
---------------  =   -----------------------------------
dE dA dOmega dt       T  * Efficiency * Aeff * BinWidth

For most experiments, BinWidth is transformed into log space for convenience:
               BinWidth = LN(10) * dlog10(E)
And then a decade wide binning is assumed: dlog10(E) = 1

But, in ANITA, they set BinWidth = 4 (!!!!!!!!!!!!!!)
See eq D1 of the ANITA-III paper.
"... the factor Delta = 4 follows the normalization convention..."

This means that the ANITA limit is a factor of LN(10)/4 too strong
when naively compared to other experiments, e.g. IceCube.
So, below, we multply by by 4/LN(10) to fix the bin width.

Regarding Auger Limits
# ====================================================

Auger publishes a limit that only applies to a single flavor.
So, to make it an all-flavor limit, the limit must be multiplied by 3.
Because this is a limit, multiplying by 3 makes the limit *weaker*.
Also, Auger uses half decade bins, so that must be corrected to a single decade.
The net factor of 3/2, on a log-log plot, leaves the limit's position (relative
to other experiments) essentially unchanged.
'''


def _load(fname, **kwargs):
    return np.loadtxt(os.path.join(local_dir, "experiments", fname), **kwargs)


def _bins_factor(fname, n, native=None):
    """
    Factor that converts a limit or sensitivity from the energy binning of its file to `n` bins per decade (see the
    convention in the module docstring). The binning of the file is taken from its header line
    "Energy bins per decade: <native>" unless `native` is given.
    """
    if native is None:
        with open(os.path.join(local_dir, "experiments", fname)) as f:
            lines = [line for line in f if line.lower().startswith("# energy bins per decade:")]
        if len(lines) != 1:
            raise ValueError(f"{fname} needs exactly one header line '# Energy bins per decade: <n>'")
        native = float(lines[0].split(":")[1])
    return n / native


@functools.lru_cache(maxsize=None)
def _load_data(n):
    """ Published data, limits and sensitivities (E^2 dN/dE in base units, limits for `n` energy bins per decade) """
    d = SimpleNamespace()

    # Other planned experiments
    # -------------------------------------------------------------------------------------------------------------
    # GRAND white paper, numerical values, Bustamante.
    # The expected sensitivities for GRAND are given for 3 years, rescaling them to 10 years
    for name in ("10k", "200k"):
        fname = f"grand{name}_sensitivity.txt"
        data = _load(fname)
        energy = data[:, 0] * units.GeV
        setattr(d, f"GRAND_{name}_energy", energy)
        setattr(d, f"GRAND_{name}", data[:, 1] * _E2_unit / energy * _bins_factor(fname, n) * 3 / 10)

    # RADAR proposed from https://arxiv.org/pdf/1710.02883.pdf
    d.Radar = _load("radar_sensitivity.txt")
    d.Radar[:, 0] = 10 ** d.Radar[:, 0] * units.eV
    d.Radar[:, 1:3] *= _E2_unit * _bins_factor("radar_sensitivity.txt", n)

    # BEACON high frequency and low frequency design, 3 years, half decade, all flavor, rescaled to 10 years
    for name in ("LF_100", "LF_1000", "HF_100", "HF_1000"):
        fname = f"beacon_{name.lower()}_sensitivity.txt"
        data = _load(fname)
        setattr(d, f"BEACON_{name}_energy", data[:, 0] * units.GeV)
        setattr(d, f"BEACON_{name}", data[:, 1] * _E2_unit * _bins_factor(fname, n) * 3 / 10)

    # From 2020 PUEO whitepaper, 30 and 100 days livetime
    for name in ("30", "100"):
        fname = f"pueo{name}_sensitivity.txt"
        data = _load(fname)
        energy = data[:, 0] * units.eV
        setattr(d, f"PUEO{name}_energy", energy)
        setattr(d, f"PUEO{name}", data[:, 1] * energy / units.GeV * _E2_unit
                * (4 / np.log(10))  # see discussion above about anita binning
                * 2.44  # convert from single event sensitivity to 90% confidence level
                * _bins_factor(fname, n))

    # TAROGE-M: 10 stations, 5 year exposure, nutau only
    # Log(Energy) GeV       Sensitivity*E^2 (GeV/cm^2 s sr)
    taroge_m = _load("taroge_m_sensitivity.txt")
    d.taroge_m_E = pow(10, taroge_m[:, 0]) * units.GeV
    d.taroge_m_flux = taroge_m[:, 1] * _E2_unit * 3.0 * _bins_factor("taroge_m_sensitivity.txt", n)

    # Published data and limits
    # -------------------------------------------------------------------------------------------------------------
    # IceCube EHE limit, Phys Rev D 98 062003 (2018), numbers private correspondence Shigeru Yoshida
    # log (E^2 * Phi [GeV cm^02 s^-1 sr^-1]) : log (E [Gev])
    d.ice_cube_limit_18 = _load("icecube_ehe18_limit.txt")
    d.ice_cube_limit_18[:, 0] = 10 ** d.ice_cube_limit_18[:, 0] * units.GeV
    d.ice_cube_limit_18[:, 1] = 10 ** d.ice_cube_limit_18[:, 1] * _E2_unit * _bins_factor("icecube_ehe18_limit.txt", n)

    # IceCube 2025 EHE paper. Data available at https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JHK49D
    d.ice_cube_limit_25 = _load("icecube_ehe25_limit.txt")
    d.ice_cube_limit_25[:, 0] *= units.GeV
    d.ice_cube_limit_25[:, 1] *= _E2_unit * _bins_factor("icecube_ehe25_limit.txt", n)

    # IceCube HESE 8 years
    # log(E (GeV)); log(E^2 dN/dE (GeV cm^-2 s-1 sr-1)); format x y -dy +dy
    hese = _load("icecube_hese_8years.txt")
    # get uncertainties in right order
    hese[:, 2] = 10 ** hese[:, 1] - 10 ** (hese[:, 1] - hese[:, 2])
    hese[:, 3] = 10 ** (hese[:, 1] + hese[:, 3]) - 10 ** hese[:, 1]
    hese[:, 0] = 10 ** hese[:, 0] * units.GeV
    hese[:, 1] = 10 ** hese[:, 1] * _E2_unit * 3
    hese[:, 2:] *= _E2_unit * 3
    d.ice_cube_hese = hese

    # IceCube nu_mu data points, 9.5 years analysis
    nu_mu = _load("icecube_numu_9.5years.txt")
    nu_mu[:, 0] = 10 ** nu_mu[:, 0] * units.GeV
    nu_mu[:, 1:] = (10 ** nu_mu[:, 1:]) * 3 * _E2_unit  # convert from single flavor to all flavor
    nu_mu[:, 2] = np.abs(nu_mu[:, 1] - nu_mu[:, 2])
    nu_mu[:, 3] = np.abs(nu_mu[:, 1] - nu_mu[:, 3])
    d.nu_mu_data = nu_mu

    # IceCube Glashow resonance event
    # Paper: https://doi.org/10.1038/s41586-021-03256-1
    # Dataset: https://doi.org/10.21234/gr2021
    # NB: the csv file gives per-flavor, but we want all flavor, so multiply by 3
    glashow = np.genfromtxt(os.path.join(local_dir, "experiments", "icecube_glashow.csv"), skip_header=2,
                            delimiter=',', names=['E_min', 'E_max', 'y', 'y_lower', 'y_upper'])
    d.i3_glashow_emin = glashow['E_min'] * units.GeV
    d.i3_glashow_emax = glashow['E_max'] * units.GeV
    for key in ('y', 'y_lower', 'y_upper'):
        setattr(d, f"i3_glashow_{key}", 3. * glashow[key] * _E2_unit * 1E-8)

    # KM3-230213A, https://www.nature.com/articles/s41586-024-08543-1
    km3 = _load("km3net25_flux.txt")
    d.KM3_230231A_E = km3[0] * units.PeV
    d.KM3_230231A_E_err = km3[1:3] * units.PeV
    d.KM3_230231A_flux = km3[3] * _E2_unit
    d.KM3_230231A_flux_err = km3[4:] * _E2_unit

    # ANITA I - III, https://arxiv.org/abs/1803.02719, Phys. Rev. D 98, 022001 (2018)
    d.anita_limit = _load("anita_i_iii_limit.txt")
    d.anita_limit[:, 0] *= units.eV
    d.anita_limit[:, 1] *= units.cm ** -2 * units.second ** -1 * units.sr ** -1
    d.anita_limit[:, 1] *= d.anita_limit[:, 0]  # convert to E^2*dN/dE
    d.anita_limit[:, 1] *= (4 / np.log(10)) * _bins_factor("anita_i_iii_limit.txt", n)  # see discussion above about strange anita binning

    # ANITA I - IV, https://arxiv.org/abs/1902.04005, Phys. Rev. D 99, 122001 (2019)
    # NB: The ANITA I-IV is indeed weaker than the ANITA I-III limit (!!)
    # The reason is not understood, but can be seen easily comparing the two limits side-by-side
    d.anita_i_iv_limit = _load("anita_i_iv_limit.txt")
    d.anita_i_iv_limit[:, 0] *= units.eV
    d.anita_i_iv_limit[:, 1] *= units.eV * units.cm ** -2 * units.second ** -1 * units.sr ** -1
    d.anita_i_iv_limit[:, 1] *= (4 / np.log(10)) * _bins_factor("anita_i_iv_limit.txt", n)  # see discussion above about strange anita binning

    # Auger neutrino limit (2019, 14.7 years), JCAP 10 (2019) 022, https://arxiv.org/abs/1906.07422
    d.auger_limit = _load("auger19_limit.txt")
    d.auger_limit[:, 0] *= units.eV
    d.auger_limit[:, 1] *= _E2_unit * 3 * _bins_factor("auger19_limit.txt", n)  # 3 flavors

    # ARA Published 2sta x 1yr analysis level limit (binning is dLogE = 1)
    d.ara_1year = _load("ara_1year_limit.txt")
    d.ara_1year[:, 0] *= units.eV
    d.ara_1year[:, 1] *= _E2_unit * _bins_factor("ara_1year_limit.txt", n)

    # Analysis from https://doi.org/10.1103/PhysRevD.102.043021  https://arxiv.org/abs/1912.00987
    # 2 stations (A2 and A3), approx 1100 days of livetime per station
    E, limit, _, _ = _load("limit_a23.txt", unpack=True)
    d.ara_4year_E = E * units.eV
    d.ara_4year_limit = limit * units.eV * units.cm ** -2 * units.second ** -1 * units.sr ** -1 * _bins_factor("limit_a23.txt", n)

    # ARA phased array 6-month limit
    E, limit, _, _ = _load("limit_ARA_PA_6months.txt", unpack=True)
    d.ara_PA_6month_E = E * units.eV
    d.ara_PA_6month_limit = limit * units.eV * units.cm ** -2 * units.second ** -1 * units.sr ** -1 * _bins_factor("limit_ARA_PA_6months.txt", n)

    # ARIANNA HRA
    d.ARIANNA_HRA = _load("arianna_hra_limit.txt")
    d.ARIANNA_HRA[:, 0] *= units.GeV
    d.ARIANNA_HRA[:, 1] *= _E2_unit * _bins_factor("arianna_hra_limit.txt", n)

    # Sensitivities of planned radio detectors
    gen2 = _load("icecube_gen2_sensitivity.txt")  # 5 years
    d.gen2_whitepaper_E = gen2[:, 0] * units.GeV
    d.gen2_whitepaper_flux = gen2[:, 1] * _E2_unit * _bins_factor("icecube_gen2_sensitivity.txt", n)
    rnog = _load("rnog_sensitivity.txt")  # 5 years
    d.RNOG_E = rnog[:, 0] * units.GeV
    d.RNOG_flux = rnog[:, 1] * _E2_unit * _bins_factor("rnog_sensitivity.txt", n)

    # diffuse gamma-ray background (Fermi): energy, E^2 dN/dE, lower and upper bound
    d.fermi_diffuse_gamma = _load("fermi_diffuse_gamma.txt")
    d.fermi_diffuse_gamma[:, 0] *= units.GeV
    d.fermi_diffuse_gamma[:, 1:] *= _E2_unit
    return d


def _load_unbinned(fname, n, energy_unit, flux_unit, rows=False):
    """
    Sensitivities whose files do not state their binning (no header line "Energy bins per decade"): raises a
    ValueError, so they can only be shown once the binning is added to the file.
    `rows`: the file holds energy and flux as two rows instead of two columns.
    """
    data = _load(fname)
    E, flux = data if rows else data.T
    return E * energy_unit, flux * flux_unit * _bins_factor(fname, n)


def _E2(energy, flux):
    """ energy and E^2 * flux in plot units """
    return energy / plotUnitsEnergy, flux * energy ** 2 / plotUnitsFlux


def get_limit_figure(diffuse=True,
                     show_ice_cube_EHE_limit=True,
                     show_ice_cube_HESE_data=True,
                     show_ice_cube_HESE_fit=True,
                     show_ice_cube_mu=True,
                     show_icecube_glashow=True,
                     show_anita_I_III_limit=False,
                     show_anita_I_IV_limit=True,
                     show_auger_limit=True,
                     show_ara=True,
                     show_ara_PA=False,
                     show_arianna=True,
                     show_grand_10k=True,
                     show_grand_200k=False,
                     show_radar=False,
                     show_Heinze=True,
                     show_TA=False,
                     show_TA_nominal=False,
                     show_TA_ICRC2021=False,
                     show_RNOG=False,
                     show_IceCubeGen2_whitepaper=False,
                     show_IceCubeGen2_ICRC2021=False,
                     show_vanVliet=True,
                     show_ara_1year=False,
                     show_prediction_arianna_200=False,
                     show_PUEO_100=False,
                     show_beacon=False,
                     show_ice_cube_EHE_limit_18=False,
                     show_KM3_230213A=True,
                     show_vanVliet_TA=False,
                     show_agn=False,
                     show_grb_afterglow=False,
                     show_blazars=False,
                     show_multimessenger=False,
                     energy_bins_per_decade=1.):
    """
    Figure of E^2 dN/dE (all flavour) with published limits, data and models.

    Parameters
    ----------
    diffuse: bool
        if True, show cosmogenic models (UHECR fits of van Vliet, Heinze, TA), else models of source populations
        (TDE, LLGRB, pulsars, clusters, NS mergers)
    show_vanVliet_TA, show_agn, show_grb_afterglow, show_blazars: bool
        show these models in addition, independent of `diffuse`
    show_multimessenger: bool
        show the diffuse gamma-ray background (Fermi) and the cosmic-ray spectrum (Auger) and extend the energy
        range down to 10 GeV, as in the multimessenger figure of the RNO-G white paper (arXiv:2010.12279)
    show_*: bool
        whether to show the respective limit, data, sensitivity or model
    energy_bins_per_decade: float
        normalisation of all limits and sensitivities, see the module docstring

    Returns
    -------
    fig, ax
    """
    d = _load_data(energy_bins_per_decade)
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    # Neutrino Models
    # Version for a diffuse flux and for a source dominated flux
    legends = []
    if diffuse:
        if show_TA:
            E, _, lower, upper = neutrino_fluxes.get_table("TA_combined_fit_m3")
            TA_m3 = ax.fill_between(*_E2(E, lower), _E2(E, upper)[1],
                                    label=r'UHECRs TA combined fit (1$\sigma$), Bergman et al.', color='C0', alpha=0.5, zorder=-1)
            legends.append(TA_m3)
        if show_TA_nominal:
            E, flux, _, _ = neutrino_fluxes.get_table("TA_combined_fit_m3")
            TA_nominal, = ax.plot(*_E2(E, flux), "k-.", label="UHECRs TA combined fit, Bergman et al.")
            legends.append(TA_nominal)
        if show_TA_ICRC2021:
            E, flux, _, _ = neutrino_fluxes.get_table("TA_ICRC2021")
            TA_ICRC2021, = ax.plot(*_E2(E, flux), "k--", label="UHECRs TA fit (ICRC2021)")
            legends.append(TA_ICRC2021)
        if show_vanVliet:
            E, vanVliet_max, _, _ = neutrino_fluxes.get_table("vanVliet2019_max")
            vanVliet_reas = neutrino_fluxes.get_table("vanVliet2019_proton10")[1]
            prot = ax.fill_between(*_E2(E, vanVliet_max), _E2(E, vanVliet_reas / 50)[1], color='0.9',
                                   label=r'allowed from UHECRs (Auger), van Vliet et al.', zorder=-2)
            legends.append(prot)
        if show_Heinze:
            E, _, lower, upper = neutrino_fluxes.get_table("Heinze2019_evolution_flat_z1")
            Auger_bestfit = ax.fill_between(*_E2(E, lower), _E2(E, upper)[1],
                                            label=r'UHECRs Auger combined fit (3$\sigma$), Heinze et al.', color='C1', alpha=0.5, zorder=1)
            legends.append(Auger_bestfit)
    else:
        E, flux, _, _ = neutrino_fluxes.get_table("FangMetzger2017_NS_merger")
        ns_merger, = ax.plot(*_E2(E, flux), color='palevioletred', label='NS-NS merger, Fang & Metzger 1707.04263', linestyle=(0, (3, 5, 1, 5)))

        E, flux, lower, upper = neutrino_fluxes.get_table("Biehl2018_TDE")
        ax.fill_between(*_E2(E, upper), _E2(E, lower)[1], color='thistle', alpha=0.5)
        p_tde, = ax.plot(*_E2(E, flux), label="TDE, Biehl et al. (1711.03555)", color='darkmagenta', linestyle=':', zorder=1)

        E, flux, lower, upper = neutrino_fluxes.get_table("Boncioli2019_LLGRB")
        ax.fill_between(*_E2(E, upper), _E2(E, lower)[1], color='0.8')
        p_ll_grb, = ax.plot(*_E2(E, flux), label="LLGRB, Boncioli et al. (1808.07481)", linestyle='-.', c='k', zorder=1)

        E, _, lower, upper = neutrino_fluxes.get_table("Fang2014_pulsars")
        p_pulsar = ax.fill_between(*_E2(E, lower), _E2(E, upper)[1], label="Pulsar, Fang et al. (1311.2044)", color='wheat', alpha=0.5)
        E, flux, _, _ = neutrino_fluxes.get_table("FangMurase2018_clusters")
        p_cluster, = ax.plot(*_E2(E, flux), label="Clusters, Fang & Murase, (1704.00015)", color="olive", zorder=1, linestyle=(0, (5, 10)))

        legends += [p_tde, p_ll_grb, p_pulsar, p_cluster, ns_merger]

    # models of the multimessenger figure in arXiv:2010.12279
    for show, model, label, style in [
            (show_vanVliet_TA, "vanVliet_TA_zmax5", "UHECRs TA composition, van Vliet et al.", dict(c='darkgoldenrod', dashes=[10, 5, 20, 5])),
            (show_agn, "Murase2014_AGN", "AGN, Murase et al. (1403.4089)", dict(c='indianred', linestyle='-.')),
            (show_grb_afterglow, "Murase2007_GRB_afterglow", "GRB afterglow, Murase (0707.1140)", dict(c='indianred', linestyle=(0, (3, 5, 1, 5)))),
            (show_blazars, "Rodrigues_blazars", "Blazars, Rodrigues et al. (2003.08392)", dict(c='indianred', dashes=[5, 7, 5, 7]))]:
        if show:
            E, flux, _, _ = neutrino_fluxes.get_table(model)
            legends += ax.plot(*_E2(E, flux), label=label, **style)

    first_legend = ax.legend(handles=legends, loc=4 if diffuse else 3, fontsize=legendfontsize, handlelength=4)
    ax.add_artist(first_legend)

    # Limits, data and sensitivities
    # -----------------------------------------------------------------------------------------------------------------
    def annotate(text, xy, **kwargs):
        ax.annotate(text, xy=xy, xycoords='data', fontsize=legendfontsize, **kwargs)

    if show_grand_10k:
        ax.plot(d.GRAND_10k_energy / plotUnitsEnergy, d.GRAND_10k / plotUnitsFlux, linestyle=":", color='saddlebrown')
        if energy_bins_per_decade == 2:
            annotate('GRAND 10k', (0.9e10 * units.GeV / plotUnitsEnergy, 2e-8),
                     horizontalalignment='left', color='saddlebrown', rotation=50)
        else:
            annotate('GRAND 10k', (1.5e19 * units.eV / plotUnitsEnergy, 3.4e-8),
                     horizontalalignment='left', va="bottom", color='saddlebrown', rotation=40)

    if show_grand_200k:
        ax.plot(d.GRAND_200k_energy / plotUnitsEnergy, d.GRAND_200k / plotUnitsFlux, linestyle=":", color='saddlebrown', lw=2)
        annotate('GRAND 200k', (1e10 * units.GeV / plotUnitsEnergy, 5e-10),
                 horizontalalignment='left', color='saddlebrown', rotation=35)

    if show_radar:
        ax.fill_between(d.Radar[:, 0] / plotUnitsEnergy, d.Radar[:, 1] / plotUnitsFlux,
                        d.Radar[:, 2] / plotUnitsFlux, facecolor='None', hatch='x', edgecolor='0.8')
        annotate('Radar', (1e9 * units.GeV / plotUnitsEnergy, 4.5e-8), horizontalalignment='left', color='0.7', rotation=45)

    if show_ice_cube_EHE_limit_18:
        ax.plot(d.ice_cube_limit_18[2:, 0] / plotUnitsEnergy, d.ice_cube_limit_18[2:, 1] / plotUnitsFlux, color='dodgerblue')
        if energy_bins_per_decade == 2:
            annotate('IceCube18', (0.6e7 * units.GeV / plotUnitsEnergy, 2e-8), horizontalalignment='center', color='dodgerblue')
        else:
            annotate('IceCube18', (3e6 * units.GeV / plotUnitsEnergy, 3e-8), horizontalalignment='center', color='dodgerblue')

    if show_ice_cube_EHE_limit:
        ax.plot(d.ice_cube_limit_25[2:, 0] / plotUnitsEnergy, d.ice_cube_limit_25[2:, 1] / plotUnitsFlux, color='dodgerblue')
        if energy_bins_per_decade == 2:
            annotate('IceCube25', (0.6e8 * units.GeV / plotUnitsEnergy, 1.5e-8), horizontalalignment='center', color='dodgerblue')
        else:
            annotate('IceCube25', (8e7 * units.GeV / plotUnitsEnergy, 7e-9), horizontalalignment='center', color='dodgerblue')

    if show_ice_cube_HESE_data:
        hese = d.ice_cube_hese
        uplimit = hese[:, 3] == 0
        ax.errorbar(hese[:, 0] / plotUnitsEnergy, hese[:, 1] / plotUnitsFlux, yerr=hese[:, 2:].T / plotUnitsFlux,
                    uplims=uplimit, color='dodgerblue', marker='o', ecolor='dodgerblue', linestyle='None', zorder=3)

    if show_ice_cube_HESE_fit:
        E = np.arange(1e5, 5e6, 1e5) * units.GeV
        lower, upper = neutrino_fluxes.get_flux_band("IceCube2017_HESE", E)
        ax.fill_between(*_E2(E, upper), _E2(E, lower)[1], hatch='//', edgecolor='dodgerblue', facecolor='azure', zorder=2)
        ax.plot(*_E2(E, neutrino_fluxes.get_flux("IceCube2017_HESE", E)), color='dodgerblue')

    if show_ice_cube_mu:
        E = np.arange(1e2, 5e6, 1e5) * units.GeV
        lower, upper = neutrino_fluxes.get_flux_band("IceCube2022_numu", E)
        ax.fill_between(*_E2(E, upper), _E2(E, lower)[1], hatch='\\', edgecolor='dodgerblue', facecolor='azure', zorder=2)
        ax.plot(*_E2(E, neutrino_fluxes.get_flux("IceCube2022_numu", E)), color='dodgerblue')
        annotate('IceCube', (3e6 * units.GeV / plotUnitsEnergy, 3e-8), horizontalalignment='center', color='dodgerblue')

        nu_mu = d.nu_mu_data
        ax.errorbar(nu_mu[:, 0] / plotUnitsEnergy, nu_mu[:, 1] / plotUnitsFlux,
                    yerr=nu_mu[:, 2:].T / plotUnitsFlux, uplims=nu_mu[:, 3] == 0, color='dodgerblue',
                    marker='o', ecolor='dodgerblue', linestyle='None', zorder=3, markersize=7)

    if show_icecube_glashow:
        # only plot the Glashow data point (the first (0) and last (2) entries are upper limits)
        point = 1
        emin, emax = d.i3_glashow_emin[point] / plotUnitsEnergy, d.i3_glashow_emax[point] / plotUnitsEnergy
        glashow_x = (emax - emin) / 2 + emin
        glashow_y = d.i3_glashow_y[point] / plotUnitsFlux
        ax.errorbar(
            x=glashow_x,
            y=glashow_y,
            xerr=[[glashow_x - emin], [emax - glashow_x]],
            yerr=[[glashow_y - d.i3_glashow_y_lower[point] / plotUnitsFlux], [d.i3_glashow_y_upper[point] / plotUnitsFlux - glashow_y]],
            marker='o', markersize=7, color='dodgerblue', ecolor='dodgerblue')

    if show_anita_I_III_limit:
        ax.plot(d.anita_limit[:, 0] / plotUnitsEnergy, d.anita_limit[:, 1] / plotUnitsFlux, color='darkorange')
        annotate('ANITA I - III', (7e9 * units.GeV / plotUnitsEnergy, 1e-6 if energy_bins_per_decade == 2 else 5e-7),
                 horizontalalignment='left', color='darkorange')

    if show_anita_I_IV_limit:
        ax.plot(d.anita_i_iv_limit[:, 0] / plotUnitsEnergy, d.anita_i_iv_limit[:, 1] / plotUnitsFlux, color='darkorange')
        if energy_bins_per_decade == 2:
            annotate('ANITA I - IV', (7e9 * units.GeV / plotUnitsEnergy, 1e-6), horizontalalignment='left', color='darkorange')
        else:
            annotate('ANITA I - IV', (1e19 * units.eV / plotUnitsEnergy, 1.2e-6), horizontalalignment='left', color='darkorange')

    if show_auger_limit:
        ax.plot(d.auger_limit[:, 0] / plotUnitsEnergy, d.auger_limit[:, 1] / plotUnitsFlux, color='forestgreen')
        if energy_bins_per_decade == 2:
            annotate('Auger', (8e16 * units.eV / plotUnitsEnergy, 2.1e-7), horizontalalignment='left', color='forestgreen')
        else:
            annotate('Auger', (1.5e17 * units.eV / plotUnitsEnergy, 7e-8), horizontalalignment='right', color='forestgreen', rotation=-45)

    if show_ara_1year:
        ax.plot(d.ara_1year[:, 0] / plotUnitsEnergy, d.ara_1year[:, 1] / plotUnitsFlux, color='indigo')
        if energy_bins_per_decade == 2:
            annotate('ARA', (5e8 * units.GeV / plotUnitsEnergy, 6e-7), horizontalalignment='left', color='indigo')
        else:
            annotate('ARA', (2e10 * units.GeV / plotUnitsEnergy, 1.05e-6), horizontalalignment='left', color='indigo')

    if show_ara:
        ax.plot(d.ara_4year_E / plotUnitsEnergy, d.ara_4year_limit / plotUnitsFlux, color='indigo')
        if energy_bins_per_decade == 2:
            annotate('ARA', (5e8 * units.GeV / plotUnitsEnergy, 6e-7), horizontalalignment='left', color='indigo')
        else:
            annotate('ARA', (1e18 * units.eV / plotUnitsEnergy, 0.7e-6), horizontalalignment='left', color='indigo')

    if show_ara_PA:
        ax.plot(d.ara_PA_6month_E / plotUnitsEnergy, d.ara_PA_6month_limit / plotUnitsFlux, color='blue', linestyle='-')
        if energy_bins_per_decade == 2:
            annotate('ARA PA', (2.95E1 * units.eV / plotUnitsEnergy, 6e-7), horizontalalignment='left', color='grey')
        else:
            annotate('ARA PA', (2.9E17 * units.eV / plotUnitsEnergy, 5.2e-6), horizontalalignment='left', color='blue')

    if show_arianna:
        ax.plot(d.ARIANNA_HRA[:, 0] / plotUnitsEnergy, d.ARIANNA_HRA[:, 1] / plotUnitsFlux, color='red')
        if energy_bins_per_decade == 2:
            annotate('ARIANNA', (5e8 * units.GeV / plotUnitsEnergy, 6e-7), horizontalalignment='left', color='red')
        else:
            annotate('ARIANNA', (1e7 * units.GeV / plotUnitsEnergy, 2e-6), horizontalalignment='left', color='red')

    if show_IceCubeGen2_whitepaper:  # 5 years
        ax.plot(d.gen2_whitepaper_E / plotUnitsEnergy, d.gen2_whitepaper_flux / 2 / plotUnitsFlux, color='purple', linestyle="--")
        annotate('IceCube-Gen2 radio', (.8e8 * units.GeV / plotUnitsEnergy, 1.6e-10), horizontalalignment='left', color='purple')

    if show_IceCubeGen2_ICRC2021:  # 10 years, https://pos.sissa.it/395/1183/
        E, flux = _load_unbinned("Gen2radio_sensitivity_ICRC2021.txt", energy_bins_per_decade, units.eV, _E2_unit,
                                 rows=True)
        ax.plot(E / plotUnitsEnergy, flux / plotUnitsFlux, color='purple', linestyle="--")
        annotate('IceCube-Gen2 radio', (.8e8 * units.GeV / plotUnitsEnergy, 1.3e-10), horizontalalignment='left', color='purple')

    if show_RNOG:
        # 5 years, uses 70% uptime from RNO-G whitepaper and rescaling to 10 years
        ax.plot(d.RNOG_E / plotUnitsEnergy, d.RNOG_flux / 0.7 / 2 / plotUnitsFlux, color='red', linestyle="-.")
        annotate('RNO-G', (8e18 * units.eV / plotUnitsEnergy, 1.5e-8), horizontalalignment='left', va="top", color='red', rotation=10)

    if show_prediction_arianna_200:
        E, flux = _load_unbinned("expected_sensivity_ARIANNA-200.txt", energy_bins_per_decade, units.GeV, _E2_unit)
        ax.plot(E / plotUnitsEnergy, flux / plotUnitsFlux, label='ARIANNA-200 (5 years)', color='blue', linestyle="--")
        annotate('ARIANNA-200', (.9e19 * units.eV / plotUnitsEnergy, 3.15e-9), horizontalalignment='left', color='blue', rotation=30)

    if show_PUEO_100:
        annotate('PUEO (3 flights)', (3e18 * units.eV / plotUnitsEnergy, 2.1e-8), horizontalalignment='left', color='#EA5A06')
        ax.plot(d.PUEO100_energy / plotUnitsEnergy, d.PUEO100 / plotUnitsFlux, linestyle=(0, (3, 1, 1, 1, 1, 1)), color='#EA5A06',
                label='PUEO (3 flights, 100 days)', lw=2)

    if show_beacon:
        ax.plot(d.BEACON_LF_1000_energy / plotUnitsEnergy, d.BEACON_LF_1000 / plotUnitsFlux,
                linestyle="-.", color='#F97807', label='BEACON 1k', lw=2)
        annotate('BEACON-1k', (7e18 * units.eV / plotUnitsEnergy, 9e-10), horizontalalignment='left', verticalalignment="bottom",
                 color='#F97807', rotation=35)

    if show_KM3_230213A:
        ax.errorbar(d.KM3_230231A_E / plotUnitsEnergy, d.KM3_230231A_flux / plotUnitsFlux,
                    yerr=d.KM3_230231A_flux_err.reshape(2, 1) / plotUnitsFlux, xerr=d.KM3_230231A_E_err.reshape(2, 1) / plotUnitsEnergy,
                    color='deeppink')
        ax.scatter(d.KM3_230231A_E / plotUnitsEnergy, d.KM3_230231A_flux / plotUnitsFlux, color='deeppink')
        annotate('KM3-230213A', (d.KM3_230231A_E / plotUnitsEnergy * 1.1, d.KM3_230231A_flux / plotUnitsFlux * 1.1), color='deeppink')

    if show_multimessenger:
        fermi = d.fermi_diffuse_gamma
        ax.errorbar(fermi[:, 0] / plotUnitsEnergy, fermi[:, 1] / plotUnitsFlux, linestyle='None', marker='<', color='firebrick',
                    yerr=np.abs(fermi[:, 2:].T - fermi[:, 1]) / plotUnitsFlux)
        annotate(r'$\gamma$-ray background' "\n" '(Fermi)', (5e1 * units.GeV / plotUnitsEnergy, 3e-7),
                 horizontalalignment='left', color='firebrick', fontweight='bold')
        E, J, J_err_low, J_err_up = cr_flux.get_measured_data('data_auger_19')
        ax.errorbar(E / plotUnitsEnergy, J * E ** 2 / plotUnitsFlux, yerr=np.array([J_err_low, J_err_up]) * E ** 2 / plotUnitsFlux,
                    linestyle='None', marker='o', color='darkorange')
        annotate('cosmic rays \n (Auger)', (2e10 * units.GeV / plotUnitsEnergy, 3e-7),
                 horizontalalignment='right', color='darkorange', fontweight='bold')

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(which='both', labelsize=14)
    ax.set_xlabel(f'neutrino energy [{plotUnitsEnergyStr}]', fontsize=18)
    ax.set_ylabel(r'$E^2\Phi$ [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]', fontsize=18)

    if diffuse:
        ax.set_ylim(1e-12, 10e-6)
    else:
        ax.set_ylim(1e-11, 2e-6)
    if show_multimessenger:
        ax.set_xlim(10 * units.GeV / plotUnitsEnergy, 4e11 * units.GeV / plotUnitsEnergy)
    else:
        ax.set_xlim(1e14 * units.eV / plotUnitsEnergy, 1e20 * units.eV / plotUnitsEnergy)

    fig.tight_layout()
    return fig, ax


def add_limit(ax, limit_labels, E, Veffsr, n_stations, label, livetime=3 * units.year, linestyle='-', color='r', linewidth=3, band=False,
              energy_bins_per_decade=1.):
    """
    Add the sensitivity (90% CL, decade energy bins) of a detector to a limit plot.

    Parameters
    ----------
    ax: matplotlib axis
        axis from :func:`get_limit_figure`
    limit_labels: list
        legend handles, the new line is appended (not for `band`)
    E: array
        energies
    Veffsr: array
        effective volume times solid angle (water equivalent); for `band`, a pair of arrays (lower, upper)
    n_stations: int
        number of stations
    label: str
        legend label, the livetime is appended
    livetime: float
    linestyle, color, linewidth:
        line style
    band: bool
        plot a band between the sensitivities of the two effective volumes in `Veffsr`
    energy_bins_per_decade: float
        normalisation of the sensitivity, must be the value used for :func:`get_limit_figure`

    Returns
    -------
    limit_labels
    """
    E = np.array(E)
    Veffsr = np.array(Veffsr)

    def get_limit(veff):
        return fluxes.get_limit_e2_flux(energy=E, veff_sr=veff, livetime=livetime, signalEff=n_stations,
                                        energyBinsPerDecade=energy_bins_per_decade, upperLimOnEvents=2.44,
                                        nuCrsScn='hedis_bgr18')

    if band:
        ax.fill_between(E / plotUnitsEnergy, get_limit(Veffsr[1]) / plotUnitsFlux, get_limit(Veffsr[0]) / plotUnitsFlux,
                        color=color, alpha=0.2)
    else:
        _plt, = ax.plot(E / plotUnitsEnergy, get_limit(Veffsr) / plotUnitsFlux, linestyle=linestyle, color=color,
                        label="{1}: {0} years".format(int(livetime / units.year), label), linewidth=linewidth)
        limit_labels.append(_plt)

    return limit_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the expected sensitivity of a detector into the limit plot.")
    parser.add_argument("veff", nargs="?", default=os.path.join(local_dir, "veffs", "punch_deep_pa_50m_veff.txt"),
                        help="text file with the columns energy [eV] and effective volume times solid angle "
                        "[km^3 sr] of one station (default: veffs/punch_deep_pa_50m_veff.txt)")
    parser.add_argument("--n-stations", type=int, nargs="+", default=[100, 1000],
                        help="number(s) of stations, one sensitivity curve each (default: %(default)s)")
    parser.add_argument("--livetime", type=float, default=5, help="livetime in years (default: %(default)s)")
    parser.add_argument("--label", default="One current design", help="legend label (default: %(default)s)")
    parser.add_argument("--bins-per-decade", type=float, default=1.,
                        help="energy bins per decade the limits and sensitivities are normalised to "
                        "(default: %(default)s)")
    parser.add_argument("--sources", action="store_true",
                        help="show models of source populations instead of cosmogenic models")
    parser.add_argument("--output", help="output file (default: Limit_diffuse.pdf or Limit_sources.pdf)")
    args = parser.parse_args()

    veff = np.loadtxt(args.veff)
    veff[:, 0] *= units.eV
    veff[:, 1] *= units.km ** 3 * units.sr

    fig, ax = get_limit_figure(diffuse=not args.sources, energy_bins_per_decade=args.bins_per_decade)
    labels = []
    for n_stations in args.n_stations:
        labels = add_limit(ax, labels, veff[:, 0], veff[:, 1], n_stations=n_stations, livetime=args.livetime * units.year,
                           label=f"{args.label}, {n_stations} stations", energy_bins_per_decade=args.bins_per_decade)
    ax.legend(handles=labels, loc=2)
    output = args.output or ("Limit_sources.pdf" if args.sources else "Limit_diffuse.pdf")
    fig.savefig(output)
    print(f"saved {output}")
