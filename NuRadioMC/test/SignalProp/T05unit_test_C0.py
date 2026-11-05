#!/usr/bin/env python3
"""
Compare the C0 parameter of the analytic ray tracing solutions for random source positions
with a reference, for several ice models. Use ``--create-reference`` to regenerate the references.
"""
import argparse
import json
import os
import numpy as np
from numpy import testing
from NuRadioMC.SignalProp import analyticraytracing as ray
from NuRadioMC.utilities import medium
from NuRadioReco.utilities import units

# ice model, maximum source depth, number of stored solutions, reference file
sites = {
    'southpole': (medium.southpole_simple_layered, -3. * units.km, 2, 'reference_C0_SP.json'),
    'mooresbay': (medium.mooresbay_simple, -0.5 * units.km, 10, 'reference_C0_MooresBay.json'),
    'greenland3exp': (medium.greenland_3exp_density, -3. * units.km, 10, 'reference_C0_greenland3exp.json'),
}


def calculate_C0s(site):
    ice_model, zmax, n_solutions, _ = sites[site]

    np.random.seed(10)  # set seed to have reproducible results
    n_events = int(1e3)
    rmin = 50. * units.m
    rmax = 3. * units.km
    zmin = 0. * units.m
    rr = np.random.triangular(rmin, rmax, rmax, n_events)
    phiphi = np.random.uniform(0, 2 * np.pi, n_events)
    zz = np.random.uniform(zmin, zmax, n_events)
    points = np.array([rr * np.cos(phiphi), rr * np.sin(phiphi), zz]).T
    x_receiver = np.array([0., 0., -5.])

    C0s = np.zeros((n_events, n_solutions))
    r = ray.ray_tracing(ice_model())
    for iX, x in enumerate(points):
        r.set_start_and_end_point(x, x_receiver)
        r.find_solutions()
        for iS in range(r.get_number_of_solutions()):
            C0s[iX, iS] = r.get_results()[iS]['C0']
    return C0s


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--site', choices=list(sites), nargs='+', default=list(sites))
parser.add_argument('--create-reference', action='store_true', help='overwrite the reference files')
args = parser.parse_args()

path = os.path.dirname(os.path.abspath(__file__))
for site in args.site:
    reference_file = os.path.join(path, sites[site][3])
    C0s = calculate_C0s(site)
    if args.create_reference:
        with open(reference_file, 'w') as fout:
            json.dump(C0s.tolist(), fout)
        print(f'Wrote {reference_file}')
        continue

    with open(reference_file, 'r') as fin:
        C0s_ref = np.array(json.load(fin))

    # rtol is loosened from the numpy default (1e-7) because the C++ raytracer relies on GSL root
    # finding, whose last-bit behavior differs slightly across platforms/GSL versions (observed
    # relative differences up to ~1.4e-7 between macOS and the reference values).
    testing.assert_allclose(C0s, C0s_ref, rtol=1e-6, err_msg=f'C0 for site {site} differs from reference')
    print(f'C0 test for site {site} passed without issues')
