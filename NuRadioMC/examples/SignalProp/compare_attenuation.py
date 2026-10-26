from NuRadioMC.SignalProp import propagation
from NuRadioMC.utilities import medium
from NuRadioReco.utilities import units

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import itertools
import argparse
import logging
import h5py
import time

"""
This script is meant to be used to compare the attenuation factor calculated with two different
methods. For example with the python implementation and the C++ implementation of the analytic
raytracer (this is currently implemented but the script could be easily modified to compare other
methods (for example comparing the optimization in the python implementation)).
See the two function `calculate_attenuation_python_cpp` and `calculate_attenuation_python_opt`.

Only one raytracing implementation is used!
"""

# Define the frequency range for the attenuation calculation
nyquist_frequency = 0.5 * units.GHz
frequencies = np.linspace(50 * units.MHz, nyquist_frequency, 100)


def calculate_attenuation_python_cpp(ray):
    """ This function takes a raytracer object and calculates the attenuation using two different
    implementations (e.g. python and cpp).

    It returns the time it took for each implementation and the results.
    It returns that for all 3 possible solution types (direct, refracted, reflected) while
    at least one is always 0 (can be all three).

    Parameters
    ----------
    ray : NuRadioMC.SignalProp.ray_tracing_modules.raytracer
        The raytracer object to use for the calculation.

    Returns
    -------
    times : np.ndarray(shape=(3, 2))
        The time it took for each implementation (e.g., cpp and python) for each solution type.
    attenuations : np.ndarray(shape=(3, 2, len(frequencies)))
        The attenuation results for each implementation (e.g., cpp and python) for each solution type.
    """

    times = [[0, 0]] * 3
    attenuations = [[np.zeros_like(frequencies), np.zeros_like(frequencies)]] * 3

    for i_solution in range(ray.get_number_of_solutions()):

        solution_int = ray.get_solution_type(i_solution)

        # This is a little hack to calculate the attenuation via the cpp implementation
        # while we use the python implementation for the raytracing
        ray._r2d.use_cpp = True
        t0 = time.time()
        attenuation_1 = ray.get_attenuation(i_solution, frequencies, nyquist_frequency)
        dt_1 = time.time() - t0

        # switch back to the python implementation
        ray._r2d.use_cpp = False
        t0 = time.time()
        attenuation_2 = ray.get_attenuation(i_solution, frequencies, nyquist_frequency)
        dt_2 = time.time() - t0

        attenuations[solution_int - 1] = [attenuation_1, attenuation_2]
        times[solution_int - 1] = [dt_1, dt_2]

    return np.asarray(times), np.asarray(attenuations)


def calculate_attenuation_python_opt(ray):
    """
    This is an alternative test (not used) of the optimization of the python implementation.

    It returns the time it took for each implementation and the results.
    It returns that for all 3 possible solution types (direct, refracted, reflected) while
    at least one is always 0 (can be all three).

    Parameters
    ----------
    ray : NuRadioMC.SignalProp.ray_tracing_modules.raytracer
        The raytracer object to use for the calculation.

    Returns
    -------
    times : np.ndarray(shape=(3, 2))
        The time it took for each implementation (e.g., cpp and python) for each solution type.
    attenuations : np.ndarray(shape=(3, 2, len(frequencies)))
        The attenuation results for each implementation (e.g., cpp and python) for each solution type.
    """

    times = [[0, 0]] * 3
    attenuations = [[np.zeros_like(frequencies), np.zeros_like(frequencies)]] * 3

    for i_solution in range(ray.get_number_of_solutions()):

        solution_int = ray.get_solution_type(i_solution)

        # hack to switch back and forth the optimized calculation
        ray._r2d._use_optimized_calculation = True
        t0 = time.time()
        attenuation_1 = ray.get_attenuation(i_solution, frequencies, nyquist_frequency)
        dt_1 = time.time() - t0

        ray._r2d._use_optimized_calculation = False
        t0 = time.time()
        attenuation_2 = ray.get_attenuation(i_solution, frequencies, nyquist_frequency)
        dt_2 = time.time() - t0

        attenuations[solution_int - 1] = [attenuation_1, attenuation_2]
        times[solution_int - 1] = [dt_1, dt_2]

    return np.asarray(times), np.asarray(attenuations)


logger = logging.getLogger('NuRadioMC.ray_tracing_modules')
logger.setLevel(logging.INFO)

### This example shows the ray tracing results for the different
### ray tracing modules available in NuRadioMC
ref_index_model = 'greenland_simple'
ice = medium.get_ice_model(ref_index_model)
attenuation_models = ['GL3']

# Specify location of receiver
final_point = np.array([0, 0, -100]) * units.m

argparser = argparse.ArgumentParser(description="Compare the attenuation of the analytic raytracer with the cpp version.")

argparser.add_argument(
    "hdf5_files",
    nargs="+",
    default=None,
    help="HDF5 files to read the initial points from. If not provided, a rectangular grid is used."
)

argparser.add_argument(
    "-n", "--n_points",
    type=float,
    default=1000,
    help="Number of points to use for the raytracing. Default: 1000"
)

args = argparser.parse_args()

# Specify source locations. Either use the default of a rectangular grid (not argument)
# or provide an NuRadioMC HDF5 file to read locations from there
if args.hdf5_files is not None:
    initial_points = []
    for hdf5_path in args.hdf5_files:
        if not hdf5_path.endswith(".hdf5"):
            raise ValueError(f"Please provide a valid HDF5 file with the ending \".hdf5\". Provided file: {hdf5_path}")

        with h5py.File(hdf5_path, "r") as f:
            event_group_ids = f["event_group_ids"]
            _, index = np.unique(event_group_ids, return_index=True)

            station_key = [key for key in f.keys() if key.startswith("station")][0]
            antenna_position = f[station_key].attrs["antenna_positions"][0]

            initial_points.append(np.array(
                [f["xx"] - antenna_position[0], f["yy"] - antenna_position[1], f["zz"]]).T[index] * units.m)

    initial_points = np.vstack(initial_points)
    if args.n_points == 0:
        args.n_points = int(len(initial_points))
    else:
        args.n_points = int(args.n_points)

    initial_points = initial_points[:args.n_points]
    n_total = len(initial_points)
    logger.info(f"Using {len(initial_points)} initial points from {hdf5_path}")
    def get_initial_points():
        for initial_point in initial_points:
            yield initial_point
else:
    n = 20
    n_total = n ** 2
    logger.info(f"Use {n_total} initial points in a rectangular grid")
    def get_initial_points():
        xs = np.linspace(-3000, -100, n)
        zs = np.linspace(-3000, -10, n)
        for x, z in itertools.product(xs, zs):
            yield np.array([x, 0, z]) * units.m


prop = propagation.get_propagation_module("analytic")

timing = []
attenuations = []
t_raytracing = 0

positions_with_solutions = []
for attenuation_model in attenuation_models:
    ray = prop(
        ice, attenuation_model,
        n_frequencies_integration=25,
        n_reflections=0, use_cpp=False, compile_numba=True,
        # ray_tracing_2D_kwards={"overwrite_speedup": False}
    )

    for idx, initial_point in enumerate(get_initial_points()):

        if idx % 100 == 0:
            print(f"Processing point {idx}")

        ray.set_start_and_end_point(initial_point, final_point)

        t0 = time.time()
        ray.find_solutions()
        t_raytracing += time.time() - t0

        times, attenuations_ray = calculate_attenuation_python_cpp(ray)
        if not ray.get_number_of_solutions():
            print(f"No solution found for point {initial_point}")
            continue

        positions_with_solutions.append(initial_point)
        attenuations.append(attenuations_ray)
        timing.append(times)

attenuations = np.array(attenuations)
timing = np.array(timing)

print(f"total time for raytracing: {t_raytracing:.2f} s")

dt = timing[:, :, 1] - timing[:, :, 0]

mask = dt != 0
dt = dt[mask]

fig, (ax, ax1) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
ax.hist(dt.flatten(), bins=30, histtype="step", label="timing difference")
ax.set_xlabel("time improvement / s (python - cpp)")
ax.legend(title=f"Number of solutions: {len(dt)}\nTotal time cpp: {np.sum(timing[:, :, 0]):.2f} s\nTotal time python: {np.sum(timing[:, :, 1]):.2f} s")

ratio = timing[:, :, 0][mask] / timing[:, :, 1][mask]
ax1.hist(ratio, bins=30, histtype="step", label="timing difference")
ax1.set_xlabel("t_cpp / t_python")
fig.tight_layout()


fig, ax = plt.subplots()
null_mask = np.squeeze(np.all([attenuations[:, :, 0] != np.zeros_like(frequencies)], axis=-1))

diff = (attenuations[:, :, 0][null_mask] / attenuations[:, :, 1][null_mask] - 1)
mask = diff > 0

ax.set_title(f"{n_total} inital points. all solution types, all frequencies")

ax.hist(diff[mask], bins=100, histtype="step", label="positive")
ax.hist(-diff[~mask], bins=100, histtype="step", label="negative")
ax.set_xlabel(r"$a_{cpp} / a_{python} - 1$")

ax.set_yscale("log")
ax.legend()
fig.tight_layout()

fig, ax = plt.subplots()
ax.set_title(f"{n_total} inital points. all solution types")

mean_ratios = []
for position, attenuation in zip(positions_with_solutions, attenuations):

    mask = np.squeeze(np.all([attenuation != np.zeros_like(frequencies)], axis=-1))[:, 0]
    attenuation = attenuation[mask]
    tmp = np.mean(attenuation[:, 0] / attenuation[:, 1], axis=-1) - 1
    mean_ratios += tmp.tolist()

lim = np.max(np.abs([np.min(mean_ratios), np.max(mean_ratios)]))

for position, attenuation in zip(positions_with_solutions, attenuations):
    # attenuation shape : 3 x 2 x 100 (num_freq)
    # 3 = number of solutions (direct, refracted, reflected) at least one should be all 0
    # 2 = methods to compare
    mask = np.squeeze(np.all([attenuation != np.zeros_like(frequencies)], axis=-1))[:, 0]
    # -> shape : 2 x 2 x 100 (num_freq)
    attenuation = attenuation[mask]
    mean_ratio = np.mean(attenuation[:, 0] / attenuation[:, 1], axis=1) - 1

    horizontal_distance = np.sqrt(position[0] ** 2 + position[1] ** 2)

    ax.scatter(
        horizontal_distance - 20, position[2],
        c=mean_ratio[0],
        s=20, vmin=-lim, vmax=lim,
        linewidth=0.2,
        edgecolors="grey", cmap="seismic"
    )
    if len(mean_ratio) > 1:
        sct = ax.scatter(
            horizontal_distance + 20, position[2],
            c=mean_ratio[1],
            linewidth=0.2,
            s=20, vmin=-lim, vmax=lim,
            edgecolors="grey", cmap="seismic"
        )

ax.scatter(
    final_point[0], final_point[2],
    c="black", s=100, marker="*"
)

cb = plt.colorbar(sct, pad=0.02)
cb.set_label(r"$\langle a_{cpp} / a_{python} - 1\rangle_{50-500MHz}$")

ax.set_xlabel("horizontal distance [m]")
ax.set_ylabel("z [m]")
fig.tight_layout()


plt.show()