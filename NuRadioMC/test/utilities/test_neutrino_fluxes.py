"""
Test of ``NuRadioMC.utilities.neutrino_fluxes``: every model loads and has a consistent table, and the fluxes at
1e18 eV agree with reference values (the interpolation is linear, so a change in a table or in the unit
conversion shows up here).

Run with ``--generate`` to print the reference dictionary for the current implementation.
"""
import sys

import numpy as np

from NuRadioMC.utilities import neutrino_fluxes
from NuRadioReco.utilities import units

RTOL = 1e-6
ENERGY = 1e18 * units.eV
E2_UNIT = units.GeV / (units.cm ** 2 * units.second * units.sr)

# E^2 dN/dE at 1e18 eV in GeV / (cm^2 s sr): [flux (None: band only), lower, upper of the band (if any)]
REFERENCE = {
    "vanVliet2019_proton10": [1.6644045461206393e-09],
    "vanVliet2019_max": [2.112044922596522e-08],
    "vanVliet_TA": [8.957727909810867e-09],
    "vanVliet_TA_zmax1": [2.4256564741487126e-09],
    "vanVliet_TA_zmax5": [6.736930560314133e-09],
    "Heinze2019_auger_best_fit": [2.2058159849806862e-12, 1.2086883166491125e-12, 4.072824242962968e-12],
    "Heinze2019_auger_best_fit_2sigma": [None, 7.834064916239887e-13, 7.12062219308956e-12],
    "Heinze2019_auger_best_fit_3sigma": [None, 2.8182790203017014e-13, 1.0117780929980533e-11],
    "Heinze2019_evolution_cutoff_z1": [None, 2.8182790203017014e-13, 1.0117780929980533e-11],
    "Heinze2019_evolution_flat_z1": [None, 2.803136604287898e-13, 1.0981685097488687e-11],
    "Heinze2019_evolution_cutoff_z3": [None, 2.8203049992781506e-13, 1.1377689324829906e-11],
    "TA_combined_fit_m3": [1.560782045005116e-08, 4.32196069145679e-09, 4.351970606492954e-08],
    "TA_ICRC2021": [8.678533255768887e-09],
    "Biehl2018_TDE": [7.81739928853634e-12, 4.028768386324707e-13, 1.8011369819313385e-11],
    "Boncioli2019_LLGRB": [1.0453459303153902e-12, 2.674048360582054e-13, 1.2468144700794442e-12],
    "Fang2014_pulsars": [None, 2.2534480009419988e-09, 1.3035336510097738e-08],
    "FangMurase2018_clusters": [4.0887194429321794e-10],
    "FangMetzger2017_NS_merger": [1.7037986160457953e-09],
    "Murase2007_GRB_afterglow": [1.408970605504324e-09],
    "Murase2014_AGN": [7.67717398350216e-09],
    "Rodrigues_blazars": [1.4618728718250004e-08],
    "IceCube2016_numu": [8.153869645085454e-09],
    "IceCube2017_numu": [5.265536511110612e-09],
    "IceCube2022_numu": [1.4304864448047927e-09, 5.116856788440587e-10, 3.84598830539797e-09],
    "IceCube2017_HESE": [1.541900544570283e-11, 7.380000000000003e-13, 2.228724369656691e-10],
}


def compute_values(model):
    """ [flux, lower, upper] of a model as E^2 dN/dE at ENERGY (None if not available) """
    out = []
    try:
        out.append(float(neutrino_fluxes.get_flux(model, ENERGY) * ENERGY ** 2 / E2_UNIT))
    except ValueError:
        out.append(None)
    try:
        lower, upper = neutrino_fluxes.get_flux_band(model, ENERGY)
        out += [float(lower * ENERGY ** 2 / E2_UNIT), float(upper * ENERGY ** 2 / E2_UNIT)]
    except ValueError:
        pass
    return out


def test_neutrino_fluxes():
    all_models = list(neutrino_fluxes.MODELS) + list(neutrino_fluxes.ICECUBE_FITS)
    assert set(REFERENCE) == set(all_models), "models and reference values differ: " \
        f"{set(REFERENCE) ^ set(all_models)}"

    for model, reference in REFERENCE.items():
        values = compute_values(model)
        assert len(values) == len(reference), f"{model}: flux/band availability changed"
        for value, ref in zip(values, reference):
            if ref is None:
                assert value is None, f"{model}: unexpectedly has a central flux"
            else:
                assert np.isclose(value, ref, rtol=RTOL, atol=0), f"{model}: {value} != {ref}"

    # tables: increasing energies, matching shapes, positive fluxes
    for model in neutrino_fluxes.MODELS:
        E, flux, lower, upper = neutrino_fluxes.get_table(model)
        assert np.all(np.diff(E) > 0), f"{model}: energies not increasing"
        for table in (flux, lower, upper):
            if table is not None:
                assert table.shape == E.shape, f"{model}: shape mismatch"
                assert np.all(table >= 0), f"{model}: negative flux"
        if lower is not None:
            # the last row of the pulsar table (1e11 GeV, far above the plotted range) has lower > upper
            n = -1 if model == "Fang2014_pulsars" else None
            assert np.all(lower[:n] <= upper[:n] * (1 + 1e-12)), f"{model}: lower band above upper band"

    # array input, and the IceCube power law: 3 flavours * 1.01e-18 at 100 TeV for IceCube2017_numu
    energies = np.array([1e17, 1e18, 1e19]) * units.eV
    assert neutrino_fluxes.get_flux("TA_ICRC2021", energies).shape == (3,)
    flux = neutrino_fluxes.get_flux("IceCube2017_numu", 100 * units.TeV)
    assert np.isclose(flux * units.GeV * units.cm ** 2 * units.second * units.sr, 3 * 1.01e-18, rtol=1e-12)

    # errors
    for call, args in [(neutrino_fluxes.get_flux, ("not_a_model", ENERGY)),
                       (neutrino_fluxes.get_flux, ("Fang2014_pulsars", ENERGY)),
                       (neutrino_fluxes.get_flux_band, ("vanVliet_TA", ENERGY)),
                       (neutrino_fluxes.get_flux_band, ("IceCube2017_numu", ENERGY))]:
        try:
            call(*args)
        except (KeyError, ValueError):
            pass
        else:
            raise AssertionError(f"{call.__name__}{args} did not raise")


if __name__ == "__main__":
    if "--generate" in sys.argv:
        for m in REFERENCE:
            print(f'    "{m}": {compute_values(m)},')
    else:
        test_neutrino_fluxes()
        print("OK")
