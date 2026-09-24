# Reconstruction tests

Pytest suite for the 3D interferometric reconstruction (`NuRadioReco/modules/interferometricDirectionReconstruction3D.py`) and the air-to-ice ray tracer, so a change to either can be checked against current behaviour before it is adopted.

## Contents

- [Files](#files)
- [Running](#running)
- [Test data](#test-data)
- [Known limitations](#known-limitations)

## Files

| File | Kind | What it checks |
|---|---|---|
| `test_tables_contract.py` | contract (fast) | NPZ keys and uniform grid, physical bounds on every travel time, in-ice tables NaN above the surface, combined table equal to the minimum over ray types bit for bit, solution-ordered tables consistent, NaN coverage of the search volume per channel |
| `test_reco_known_answer.py` | known answer | exact recovery (0.2 degree, 2 m) on five geometries the current configuration resolves exactly at SNR 50; an accuracy gate over 16 sources at SNR 20; two-table multiray and polarization-group configurations; saved-peak ordering |
| `test_reco_invariances.py` | property | global time shift, amplitude scale, channel insertion order, azimuth rotation, SNR degradation, no confident peak on pure noise |
| `test_golden_master.py` | golden master | eight seeded synthetic events compared bit-tight (1e-6) to a stored reference, plus an accuracy gate |
| `test_air_ice_tracer.py` | property | the air-to-ice tracer against an independent Fermat minimisation, Snell's law at the entry point, travel-time bounds, continuity across the surface inside the exit cone, the `air_ice` propagator |
| `test_above_surface_reco.py` | known answer | reconstruction of sources above the surface with air-ice tables |
| `synthetic.py` | helper | builds events whose traces carry a band-limited pulse at the table travel times of a chosen source, using the reconstruction's own table loader and antenna geometry |
| `conftest.py` | helper | fixtures, the reference reconstruction configuration, the `slow` and `airice` markers |
| `golden/st23_synthetic_golden.json` | reference | stored output for `test_golden_master.py` |

The synthetic events test the search machinery against exactly the geometry it assumes, with no simulation in the loop. The accuracy gates in `test_reco_known_answer.py` and `test_golden_master.py` sit at the level measured on the current code (median 1.05 degree, 68th percentile 1.69 degree, 88 percent within 3 degrees over the 16-source set at SNR 20), with margin; the gate values are in the tests. They are measured rather than fixed because the current search settles 1 to 3.5 degrees off on several ideal geometries at any SNR, and about 30 degrees off on one deep source. A change that improves this passes; one that worsens it fails.

## Running

Run from this directory with the NuRadioMC checkout under test importable (installed, or its root on `PYTHONPATH`):

```bash
python -m pytest -q                      # everything
python -m pytest -q -m "not slow"        # contract tests only, seconds
python -m pytest -q test_golden_master.py
```

The reconstruction tests compile numba kernels on first use and take about a minute in total.

Inputs are set with environment variables:

| Variable | Used by | Content |
|---|---|---|
| `RECO3D_TEST_TABLES` | all reconstruction tests | root of the in-ice travel-time tables (`station{N}/` subdirectories); defaults to `NURADIO_TABLE_DIR` |
| `RECO3D_TEST_DETECTOR_FILE` | all reconstruction tests | exported detector description to use instead of the RNO-G database (station 23 at 2022-10-01) |
| `RECO3D_TEST_AIRICE_TABLES` | `test_above_surface_reco.py` | root of the air-ice tables; no default |
| `RECO3D_REGEN_GOLDEN=1` | `test_golden_master.py` | rewrite the golden reference from the imported code instead of comparing |

Tests that cannot find their inputs skip rather than fail.

To regenerate the golden reference, make the reference version of the code importable and run:

```bash
RECO3D_REGEN_GOLDEN=1 python -m pytest -q test_golden_master.py
```

## Test data

- All events are synthetic; no test reads recorded data.
- The travel-time tables are read from disk and are not part of the repository; the detector description comes from the RNO-G database or a file.
- Regenerate the golden reference only when a change to in-ice behaviour is intended, and say why in the commit.

## Known limitations

- Without `RECO3D_TEST_DETECTOR_FILE` the tests query the RNO-G database, which needs network access and database credentials.
- Reference numbers are for station 23 with greenland_simple tables and the 2022-10-01 detector description; another station or table set needs new exact-recovery geometries and gate values.
- The 1e-6 golden tolerance was set in one software environment; a different numpy or numba build can move results at that level. Check that the accuracy gates still pass, then regenerate.
- `test_air_ice_tracer.py` and `test_above_surface_reco.py` need the air-ice tracer and its propagator; on a checkout without them they skip.
