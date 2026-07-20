#!/usr/bin/env python3

import numpy as np
import pandas as pd
import pytest

import pprint

from NuRadioReco.utilities import units
from NuRadioMC.utilities import medium
from NuRadioMC.SignalProp import propagation

# Import your existing functions from the comparison script
from NuRadioMC.SignalProp.AnalyticRayTracingImpl.MultilayerAnalyticRayTracing.testutils import (
    make_grid,
    run_batch_comparison_full,
    flatten_full,
)

GENERATE_REFERENCE = False


END_POINT = np.array([0., 0., -100.013]) * units.m


GRID_FINE = dict(
    x_range=(-800.0, -0.11),
    z_range=(-1000.0, 300.0),
    n_x=100,
    n_z=100,
)

GRID_COARSE = dict(
    x_range=(-800.0, -0.11),
    z_range=(-1000.0, 300.0),
    n_x=10,
    n_z=10,
)


TEST_CASES = [

    dict(
        name="simple_vs_simple_layered",
        module_a="analytic",
        ice_a="greenland_simple",
        module_b="analytic",
        ice_b="greenland_simple_layered",
        grid=GRID_FINE,
    )
]

''',
dict(
    name="simple_layered_vs_3exp_layered",
    module_a="analytic",
    ice_a="greenland_simple_nils_layered",
    module_b="analytic",
    ice_b="greenland_3exp_nils_layered",
    grid=GRID_FINE,
)'''

    #dict(
    #    name="firn_layered_vs_radiopropa",
    #    module_a="analytic",
    #    ice_a="greenland_firn_layered",
    #    module_b="radiopropa",
    #    ice_b="greenland_firn",
    #    grid=GRID_COARSE,
    #),

REFERENCE = {

    "simple_vs_simple_layered": {

        "time_diff": {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "max_abs": 0.0,
        },

        "path_diff": {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "max_abs": 0.0,
        },

        "angle_diff": {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "max_abs": 0.0,
        },

    },

    "simple_layered_vs_3exp_layered": {},

    "firn_layered_vs_radiopropa": {},

}


def make_tracer(module_name, ice_model):

    if module_name == "analytic":

        ice = medium.get_ice_model(ice_model)
        mode = "analytic"

    elif module_name == "radiopropa":

        ice = medium.get_ice_model(ice_model)
        mode = "radiopropa"

    else:
        raise ValueError(module_name)

    prop = propagation.get_propagation_module(mode)

    return prop(
        ice,
        attenuation_model="GL1",
        n_frequencies_integration=25,
        n_reflections=0,
    )


def evaluate_statistics(df):

    stats = {}

    for quantity in [

        "time_diff",
        "time_diff_rel",
        "path_diff",
        "angle_diff",

    ]:

        if quantity not in df.columns:
            continue

        vals = df[quantity].dropna()

        stats[quantity] = {

            "mean": float(vals.mean()),
            "median": float(vals.median()),
            "std": float(vals.std()),
            "max_abs": float(np.max(np.abs(vals))),

        }

    return stats


def print_statistics(name, stats):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(pd.DataFrame(stats).T)

    print()


def assert_statistics(stats, reference):

    for quantity in reference:

        for stat in reference[quantity]:

            np.testing.assert_allclose(

                stats[quantity][stat],
                reference[quantity][stat],

                rtol=1e-4,
                atol=1e-2,

                err_msg=f"{quantity}: {stat}"

            )


TIME_TOL = 1e-4     
PATH_TOL = 1e-4
ANGLE_TOL = 1e-4


@pytest.mark.parametrize("case", TEST_CASES)
def test_raytracing(case):

    tracer_a = make_tracer(
        case["module_a"],
        case["ice_a"],
    )

    tracer_b = make_tracer(
        case["module_b"],
        case["ice_b"],
    )

    grid, _, _ = make_grid(**case["grid"])

    results = run_batch_comparison_full(
        tracer_a,
        tracer_b,
        grid,
        END_POINT,
    )

    df = pd.DataFrame(flatten_full(results))


    df["time_diff_rel"] = np.abs(df["time_diff"] / df["time_a"])
    df["path_diff_rel"] = np.abs(df["path_diff"] / df["path_a"])
    df["angle_diff_rel"] = np.abs(df["angle_diff"] / df["receive_angle_a"])
    df["attenuation_diff_rel"] = np.abs(df["attenuation_diff"] / df["attenuation_a"])
    df["focusing_diff_rel"] = np.abs(df["focusing_diff"] / df["focusing_a"])

    assert np.all(df["time_diff_rel"].dropna() < TIME_TOL)
    assert np.all(df["path_diff_rel"].dropna() < PATH_TOL)
    assert np.all(df["angle_diff_rel"].dropna() < ANGLE_TOL)

    #stats = evaluate_statistics(df)

    #print_statistics(case["name"], stats)
    if False: # GENERATE_REFERENCE:
        print(case["name"])
        pprint.pprint(stats)
        return
<<<<<<< HEAD

    elif REFERENCE[case["name"]]:
=======
    
    elif False: #REFERENCE[case["name"]]:
>>>>>>> edecd6dbd (New tests and adapting init for multilayer media)
        assert_statistics(
            stats,
            REFERENCE[case["name"]],
        )