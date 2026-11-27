#!/usr/bin/env python3
"""
Compare the numba and the python implementation of the ARZ vector potential.

``ARZ.get_vector_potential_numba`` is optimised for numba, ``ARZ.get_vector_potential`` is the
vectorised python implementation (used without numba). Both have to agree up to floating point
rounding for a grid of shower types, energies, viewing angles, distances, trace lengths and
interpolation factors. Exits with a non-zero code on failure.
"""
import sys
import numpy as np

from NuRadioReco.utilities import units
from NuRadioMC.SignalGen.ARZ import ARZ

#: maximal allowed deviation relative to the maximum of the vector potential
RTOL = 1e-10

if not ARZ.numba_available:
    print("numba is not available (or disabled), nothing to compare.")
    sys.exit(1)

arz = ARZ.ARZ(arz_version='ARZ2020', seed=1)
model_parameters = {
    "HAD": dict(Af=arz._Af_p, t0_pos=arz._t0_p_pos, freq_pos=arz._freq_p_pos, exp_pos=arz._exp_p_pos,
                t0_neg=arz._t0_p_neg, freq_neg=arz._freq_p_neg, exp_neg=arz._exp_p_neg),
    "EM": dict(Af=arz._Af_e, t0_pos=arz._t0_e_pos, freq_pos=arz._freq_e_pos, exp_pos=arz._exp_e_pos,
               t0_neg=arz._t0_e_neg, freq_neg=arz._freq_e_neg, exp_neg=arz._exp_e_neg),
}

n_index = 1.78
cherenkov_angle = np.arccos(1. / n_index)

n_tested = 0
failures = []
for shower_type in ["HAD", "EM"]:
    for energy in [1e16 * units.eV, 1e19 * units.eV]:
        profile_depth, profile_ce = arz.get_shower_profile(energy, shower_type, 3)
        for delta_theta in [0, 0.3, 3, 15]:
            for distance in [50 * units.m, 2000 * units.m]:
                for n_samples, dt in [(256, 0.5 * units.ns), (1280, 0.2 * units.ns)]:
                    for interp_factor, interp_factor2, shift_for_xmax in [(1, 100, False), (1, 1, False),
                                                                          (3, 20, True)]:
                        kwargs = dict(
                            shower_type=shower_type, n_index=n_index, distance=distance,
                            interp_factor=interp_factor, interp_factor2=interp_factor2,
                            shift_for_xmax=shift_for_xmax, em_factor=0.9, **model_parameters[shower_type])
                        args = (energy, cherenkov_angle + delta_theta * units.deg, n_samples, dt,
                                profile_depth, profile_ce)

                        vp_python = ARZ.get_vector_potential(*args, **kwargs)
                        vp_numba = ARZ.get_vector_potential_numba(*args, **kwargs)
                        n_tested += 1

                        scale = np.max(np.abs(vp_python))
                        if vp_python.shape != vp_numba.shape or scale == 0:
                            failures.append(f"{shower_type} E={energy:.0e} dtheta={delta_theta} R={distance} "
                                            f"N={n_samples}: shape {vp_numba.shape} vs. {vp_python.shape}, "
                                            f"max={scale}")
                            continue

                        deviation = np.max(np.abs(vp_numba - vp_python)) / scale
                        if not deviation <= RTOL:
                            failures.append(f"{shower_type} E={energy:.0e} dtheta={delta_theta} R={distance} "
                                            f"N={n_samples} interp={interp_factor},{interp_factor2}: "
                                            f"relative deviation {deviation:.2e}")

if failures:
    print(f"{len(failures)} of {n_tested} cases differ:\n\t" + "\n\t".join(failures))
    sys.exit(1)
print(f"numba and python implementation agree in all {n_tested} cases (rtol = {RTOL}).")
