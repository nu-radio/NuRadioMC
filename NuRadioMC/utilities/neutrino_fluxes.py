"""
Neutrino flux models: measured astrophysical fluxes, cosmogenic (UHECR) models and source populations.

All fluxes are all-flavour differential fluxes dN/dE in NuRadio base units (1 / (energy area time solid angle)).
Tabulated models are interpolated linearly and extrapolated outside the tabulated range; their tables are in
``NuRadioMC/utilities/data/neutrino_fluxes``.

Use :func:`get_flux` and :func:`get_flux_band` with a model name from :data:`MODELS`, e.g.::

    from NuRadioMC.utilities import neutrino_fluxes
    flux = neutrino_fluxes.get_flux("vanVliet2019_proton10", energies)
    lower, upper = neutrino_fluxes.get_flux_band("Heinze2019_auger_best_fit_3sigma", energies)
"""
import functools
import os

import numpy as np
from scipy.interpolate import interp1d

from NuRadioReco.utilities import units

_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "neutrino_fluxes")
_E2_flux_unit = units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1  # unit of E^2 dN/dE in most tables
_flux_unit = units.GeV ** -1 * units.cm ** -2 * units.second ** -1 * units.sr ** -1


# IceCube single power-law fits of the astrophysical flux, given as (spectral index, per-flavour normalisation at
# 100 TeV in 1e-18 / (GeV cm^2 s sr)) and, for bands, the parameter pairs spanning the band.
ICECUBE_FITS = {
    # https://arxiv.org/abs/1607.08006
    "IceCube2016_numu": dict(slope=-2.13, offset=0.9),
    # https://doi.org/10.22323/1.301.1005 (ICRC2017)
    "IceCube2017_numu": dict(slope=-2.19, offset=1.01),
    # 9.5 years through-going muons, https://arxiv.org/abs/2111.10299
    "IceCube2022_numu": dict(
        slope=-2.37, offset=1.44,
        upper=[dict(slope=-2.46, offset=1.69), dict(slope=-2.28, offset=1.69)],
        lower=[dict(slope=-2.46, offset=1.18), dict(slope=-2.28, offset=1.18)]),
    # HESE 6 years, PoS(ICRC2017)981
    "IceCube2017_HESE": dict(
        slope=-2.92, offset=2.46,
        upper=[dict(slope=-2.63, offset=2.46), dict(slope=-2.92, offset=2.76)],
        lower=[dict(slope=-3.25, offset=2.46), dict(slope=-2.92, offset=2.16)]),
}


def ice_cube_power_law(energy, slope, offset):
    """
    All-flavour flux of a single power law with IceCube's per-flavour normalisation `offset`
    (in 1e-18 / (GeV cm^2 s sr) at 100 TeV) and spectral index `slope`.
    """
    return 3 * offset * 1e-18 * (energy / (100 * units.TeV)) ** slope * _flux_unit


def _load_E2(fname, energy_col=0, flux_cols=(1,), rows=False, log10=False, factor=1, delimiter=None):
    """ Load a table of E [GeV] and E^2 dN/dE [GeV cm^-2 s^-1 sr^-1] and return E and dN/dE in base units """
    data = np.loadtxt(os.path.join(_data_dir, fname), delimiter=delimiter)
    if rows:
        data = data.T
    if log10:
        data = 10 ** data
    energy = data[:, energy_col] * units.GeV
    return (energy, *[data[:, col] * factor * _E2_flux_unit / energy ** 2 for col in flux_cols])


def _load_heinze(fname, cols):
    """ Heinze et al. tables give E [GeV] and dN/dE [GeV^-1 cm^-2 s^-1 sr^-1] """
    data = np.loadtxt(os.path.join(_data_dir, fname))
    return (data[:, 0] * units.GeV, *[data[:, col] * _flux_unit for col in cols])


def _load_vanVliet_max():
    energy, flux_1 = _load_E2("MaxNeutrinos1.txt", rows=True)
    energy_2, flux_2 = _load_E2("MaxNeutrinos2.txt", rows=True)
    assert np.allclose(energy, energy_2)
    return energy, np.maximum(flux_1, flux_2)


def _load_TA_m3():
    energy, flux = _load_E2("TA_combined_fit_m3.txt")
    energy_a, flux_a = _load_E2("TA_combined_fit_low_exp_uncertainty.txt")
    energy_b, flux_b = _load_E2("TA_combined_fit_high_exp_uncertainty.txt")
    assert np.allclose(energy, energy_a) and np.allclose(energy, energy_b)
    return energy, flux, np.minimum(flux_a, flux_b), np.maximum(flux_a, flux_b)


# name: (description with reference, loader returning (energy, flux) or (energy, lower, upper) or
# (energy, flux, lower, upper))
MODELS = {
    # cosmogenic neutrinos from UHECR fits
    "vanVliet2019_proton10": (
        "10% protons at the source (Auger), van Vliet et al., Phys. Rev. D 100, 021302 (2019), arXiv:1901.01899",
        lambda: _load_E2("ReasonableNeutrinos1.txt", rows=True)),
    "vanVliet2019_max": (
        "maximum flux allowed by UHECR data (Auger), van Vliet et al., Phys. Rev. D 100, 021302 (2019)",
        _load_vanVliet_max),
    "vanVliet_TA": (
        "TA best-fit composition, scenario A, van Vliet et al. (used in arXiv:2010.12279)",
        lambda: _load_E2("NuSpectrum_TABestFit.txt", rows=True)),
    "vanVliet_TA_zmax1": (
        "TA best-fit composition, scenario A, zmax = 1, van Vliet et al. (used in arXiv:2010.12279)",
        lambda: _load_E2("NuSpectrum_TABestFit_scenA_z=1_header.txt", rows=True)),
    "vanVliet_TA_zmax5": (
        "TA best-fit composition, scenario A, zmax = 5, van Vliet et al. (used in arXiv:2010.12279)",
        lambda: _load_E2("NuSpectrum_TABestFit_scenA_z=5_header.txt", rows=True)),
    "Heinze2019_auger_best_fit": (
        "best fit to Auger UHECR data with 1 sigma band, Heinze et al., arXiv:1901.03338",
        lambda: _load_heinze("talys_neu_bands.out", cols=(1, 2, 3))),
    "Heinze2019_auger_best_fit_2sigma": (
        "2 sigma band of Heinze2019_auger_best_fit",
        lambda: _load_heinze("talys_neu_bands.out", cols=(4, 5))),
    "Heinze2019_auger_best_fit_3sigma": (
        "3 sigma band of Heinze2019_auger_best_fit",
        lambda: _load_heinze("talys_neu_bands.out", cols=(6, 7))),
    "Heinze2019_evolution_cutoff_z1": (
        "3 sigma band for a source evolution with cutoff at z = 1, Heinze et al. (2019)",
        lambda: _load_heinze("talys_neu_evolutions.out", cols=(1, 2))),
    "Heinze2019_evolution_flat_z1": (
        "3 sigma band for a source evolution flat above z = 1, Heinze et al. (2019)",
        lambda: _load_heinze("talys_neu_evolutions.out", cols=(3, 4))),
    "Heinze2019_evolution_cutoff_z3": (
        "3 sigma band for a source evolution with cutoff at z = 3, Heinze et al. (2019)",
        lambda: _load_heinze("talys_neu_evolutions.out", cols=(5, 6))),
    "TA_combined_fit_m3": (
        "combined fit to TA energy and Xmax data (gamma = 2.0, Rcut = 1514 EeV, m = 3), D. Bergman, private "
        "communication; band: experimental uncertainty",
        _load_TA_m3),
    "TA_ICRC2021": (
        "GZK prediction from the fit to TA data, PoS(ICRC2021)338, https://pos.sissa.it/395/338/",
        lambda: _load_E2("TA_GZKprediction_ICRC2021.txt")),
    # source populations
    "Biehl2018_TDE": (
        "tidal disruption events, Biehl et al., arXiv:1711.03555 (best fit, band: min/max)",
        lambda: _load_E2("TDEneutrinos.txt", flux_cols=(1, 3, 2), factor=3)),  # tables are single flavour
    "Boncioli2019_LLGRB": (
        "low-luminosity GRBs, Boncioli et al., arXiv:1808.07481 (best fit, band: min/max)",
        lambda: _load_E2("LLGRBneutrinos.txt", flux_cols=(1, 3, 2), factor=3)),  # tables are single flavour
    "Fang2014_pulsars": (
        "pulsars, Fang et al., arXiv:1311.2044 (band only)",
        lambda: _load_E2("Pulsar_Fang+_2014.txt", flux_cols=(1, 2))),
    "FangMurase2018_clusters": (
        "galaxy clusters, Fang & Murase, arXiv:1704.00015",
        lambda: _load_E2("cluster_Fang_Murase_2018.txt")),
    "FangMetzger2017_NS_merger": (
        "neutron-star mergers, Fang & Metzger, arXiv:1707.04263",
        lambda: _load_E2("ns_merger_Fang_Metzger.txt")),
    "Murase2007_GRB_afterglow": (
        "GRB afterglows (wind, optimistic), Murase, arXiv:0707.1140",
        lambda: _load_E2("grb_ag_wind_optimistic.dat", log10=True)),
    "Murase2014_AGN": (
        "AGN, Murase et al., arXiv:1403.4089",
        lambda: _load_E2("agn_murase.dat", delimiter=",")),
    "Rodrigues_blazars": (
        "blazars, Rodrigues et al., arXiv:2003.08392",
        lambda: _load_E2("blazars_Rodrigues.txt")),
}


@functools.lru_cache(maxsize=None)
def get_table(model):
    """
    Tabulated values of a model.

    Parameters
    ----------
    model: str
        model name, a key of :data:`MODELS`

    Returns
    -------
    energy: array
    flux: array or None
        dN/dE, None for models that only provide a band
    lower, upper: arrays or None
        dN/dE of the band, None for models without band
    """
    if model not in MODELS:
        raise KeyError(f"Unknown neutrino flux model '{model}'. Available: {', '.join(MODELS)}")
    table = MODELS[model][1]()
    if len(table) == 2:
        return table[0], table[1], None, None
    if len(table) == 3:
        return table[0], None, table[1], table[2]
    return table


def get_flux(model, energy):
    """
    All-flavour flux dN/dE of a model at `energy`.

    Parameters
    ----------
    model: str
        a key of :data:`MODELS` or of :data:`ICECUBE_FITS`
    energy: float or array
        neutrino energy
    """
    if model in ICECUBE_FITS:
        return ice_cube_power_law(energy, ICECUBE_FITS[model]["slope"], ICECUBE_FITS[model]["offset"])
    E, flux, _, _ = get_table(model)
    if flux is None:
        raise ValueError(f"Model '{model}' only provides a band, use get_flux_band")
    return interp1d(E, flux, bounds_error=False, fill_value="extrapolate")(energy)


def get_flux_band(model, energy):
    """
    Lower and upper all-flavour flux dN/dE of a model's band at `energy`.

    Parameters
    ----------
    model: str
        a key of :data:`MODELS` or of :data:`ICECUBE_FITS`
    energy: float or array
        neutrino energy

    Returns
    -------
    lower, upper
    """
    if model in ICECUBE_FITS:
        fit = ICECUBE_FITS[model]
        if "upper" not in fit:
            raise ValueError(f"IceCube fit '{model}' has no band")
        upper = np.maximum(*[ice_cube_power_law(energy, **p) for p in fit["upper"]])
        lower = np.minimum(*[ice_cube_power_law(energy, **p) for p in fit["lower"]])
        return lower, upper
    E, _, lower, upper = get_table(model)
    if lower is None:
        raise ValueError(f"Model '{model}' has no band")
    return (interp1d(E, lower, bounds_error=False, fill_value="extrapolate")(energy),
            interp1d(E, upper, bounds_error=False, fill_value="extrapolate")(energy))
