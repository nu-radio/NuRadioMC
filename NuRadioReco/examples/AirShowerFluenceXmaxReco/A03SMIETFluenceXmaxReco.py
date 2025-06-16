import os
import glob
import logging
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as opt

import NuRadioReco.framework.event
import NuRadioReco.framework.base_trace
import NuRadioReco.modules.electricFieldBandPassFilter

from NuRadioReco.utilities import units
from NuRadioReco.modules.io.coreas import coreas
from NuRadioReco.modules.template_synthesis.smietSynthesis import (
    smietSynthesis,
    smietInterpolated,
)
from NuRadioReco.utilities.dataservers import download_from_dataserver
from NuRadioReco.framework.parameters import showerParameters as shp

from smiet.numpy import Shower


# Create logger
logger = logging.getLogger("NuRadioReco.SMIETFluenceXmaxReco")
logger.setLevel(10)

# Create directory for plots
if not os.path.exists("plots"):
    os.makedirs("plots")

# Download coreas simulations, if they are not present in the directory
remote_path = "data/CoREAS/LOFAR/evt_00001"
path = "data/CoREAS/LOFAR/evt_00001"
for i in range(30):
    if i == 8 or i == 20:
        # These do not exist on the remote
        continue
    to_download = os.path.join(remote_path, f"SIM0000{i:02d}.hdf5")
    if not os.path.exists(to_download):
        download_from_dataserver(
            os.path.join(remote_path, f"SIM0000{i:02d}.hdf5"),
            os.path.join(path, f"SIM0000{i:02d}.hdf5"),
        )


# Define some functions for later
def calculate_fluence_around_peak(
    efield: NuRadioReco.framework.base_trace.BaseTrace,
    signal_window: float = 25 * units.ns,
    sample_axis: int = 1,
):
    sampling: float = efield.get_sampling_rate()
    trace = efield.get_trace()

    peak_sample = np.argmax(trace, axis=sample_axis)
    window_lower = np.clip(
        peak_sample - int(signal_window / sampling), 0, trace.shape[sample_axis]
    )
    window_higher = np.clip(
        peak_sample + int(signal_window / sampling), 0, trace.shape[sample_axis]
    )

    fluence = []
    for signal, low, high in zip(trace, window_lower, window_higher):
        fluence.append(np.sum(signal[low:high] ** 2))

    return np.asarray(fluence)


def sort_array(
    my_array: np.ndarray, sort_array: np.ndarray | None = None
) -> np.ndarray:
    if sort_array is None:
        sort_array = my_array

    sort_first_ax = np.argsort(sort_array[:, 1])
    sort_zero_ax = np.argsort(sort_array[sort_first_ax, 0])

    return my_array[sort_first_ax][sort_zero_ax]


def mean_square_error(
    evt_data: NuRadioReco.framework.event.Event,
    evt_sim: NuRadioReco.framework.event.Event,
    amplitude_scale: float,
):
    data_fluences = []
    data_positions = []
    for efield in evt_data.get_station().get_sim_station().get_electric_fields():
        data_fluences.append(np.sum(calculate_fluence_around_peak(efield)))
        data_positions.append(efield.get_position())
    data_fluences = np.asarray(data_fluences)

    sim_fluences = []
    sim_positions = []
    for efield in evt_sim.get_station().get_sim_station().get_electric_fields():
        sim_fluences.append(np.sum(calculate_fluence_around_peak(efield)))
        sim_positions.append(np.squeeze(efield.get_position()))
    sim_fluences = np.asarray(sim_fluences)

    # Sort the arrays by position, to ensure matching same antennas
    data_fluences_sorted = sort_array(data_fluences, np.asarray(data_positions))
    sim_fluences_sorted = sort_array(sim_fluences, np.asarray(sim_positions))

    mse = np.sum((data_fluences_sorted - amplitude_scale * sim_fluences_sorted) ** 2)

    return mse


def mean_square_error_vvB(
    evt_data: NuRadioReco.framework.event.Event,
    evt_sim: NuRadioReco.framework.event.Event,
    amplitude_scale: float,
):
    data_fluences = []
    data_positions = []
    for efield in evt_data.get_station().get_sim_station().get_electric_fields():
        # Take only antennas on the vvB axis
        efield_pos_showerplane = (
            evt_data.get_first_sim_shower()
            .get_coordinatesystem()
            .transform_to_vxB_vxvxB(efield.get_position())
        )
        if abs(efield_pos_showerplane[0]) < 1e-3:
            logger.debug(f"Adding field at position {efield_pos_showerplane}")
            data_fluences.append(np.sum(calculate_fluence_around_peak(efield)))
            data_positions.append(efield_pos_showerplane)

    data_fluences = np.asarray(data_fluences)

    sim_fluences = []
    sim_positions = []
    for efield in evt_sim.get_station().get_sim_station().get_electric_fields():
        efield_pos_showerplane = (
            evt_data.get_first_sim_shower()
            .get_coordinatesystem()
            .transform_to_vxB_vxvxB(efield.get_position())
        )
        if abs(efield_pos_showerplane[0]) < 1e-3:
            logger.debug(f"Adding field at position {efield_pos_showerplane}")
            sim_fluences.append(np.sum(calculate_fluence_around_peak(efield)))
            sim_positions.append(efield_pos_showerplane)

    sim_fluences = np.asarray(sim_fluences)

    # Sort the arrays by position, to ensure matching same antennas
    data_fluences_sorted = sort_array(data_fluences, np.asarray(data_positions))
    sim_fluences_sorted = sort_array(sim_fluences, np.asarray(sim_positions))

    mse = np.sum((data_fluences_sorted - amplitude_scale * sim_fluences_sorted) ** 2)

    return mse


def gaisser_hillas(X, Nmax, Xmax, L, R):
    Xprime = X - Xmax
    t1 = 1 + R * Xprime / L
    pow = R**-2
    exp = -Xprime / (L * R)

    N = Nmax * t1**pow * np.exp(exp)
    N = np.nan_to_num(N)
    N = np.where(N < 1, 1, N)

    return np.stack((X, N), axis=1)


# Initialize the modules
efieldBandpassFilter = (
    NuRadioReco.modules.electricFieldBandPassFilter.electricFieldBandPassFilter()
)

# Filter settings
filter_settings = {
    "passband": [30 * units.MHz, 80 * units.MHz],
    "filter_type": "butter",
    "order": 10,
}

# Read in the "data" shower and filter the electric fields
input_file = os.path.join(path, "SIM000000.hdf5")
mc_dreamland_data = coreas.read_CORSIKA7(input_file)

mc_dreamland_xmax = mc_dreamland_data.get_first_sim_shower().get_parameter(
    shp.shower_maximum
)

efieldBandpassFilter.run(
    mc_dreamland_data,
    mc_dreamland_data.get_station().get_sim_station(),
    None,
    **filter_settings,
)

# Find all other simulations available to construct parabola
input_files = glob.glob(os.path.join(path, "*.hdf5"))
N_showers = len(input_files)

# Initialize arrays to store the fit results
fmin = np.zeros(N_showers)
success = np.zeros(N_showers, dtype=bool)
Xmax = np.zeros(N_showers)
energy_factors = np.zeros(N_showers)

# Loop over all CoREAS simulations and compute chi2
# for i, filename in enumerate(input_files):
#     # skip the CoREAS simulation that was used to generate the event under test
#     if filename == input_file:
#         continue
#
#     # read in CoREAS simulation
#     current_sim: NuRadioReco.framework.event.Event = coreas.read_CORSIKA7(filename)
#     Xmax[i] = current_sim.get_first_sim_shower()[shp.shower_maximum]
#
#     # filter the simulation with the same settings as before
#     efieldBandpassFilter.run(
#         current_sim,
#         current_sim.get_station().get_sim_station(),
#         None,
#         **filter_settings,
#     )
#
#     def obj(x):
#         return mean_square_error(mc_dreamland_data, current_sim, x[0])
#
#     res = opt.minimize(obj, [1], method="Nelder-Mead")
#
#     fmin[i] = res.fun
#     success[i] = res.success
#     energy_factors[i] = res.x[0]
#     logger.info(
#         f"Fitting footprint{i} from file {filename} gave the following result: {res}"
#     )

# SMIET fitting
input_files = glob.glob(
    "/home/mitjadesmet/Data/Showers_for_Xmax_reco/vertical_geometry/*.hdf5"
)
template_files = [file[:-4] + "npz" for file in input_files]

N_showers = len(input_files)
Xmax_smiet = np.zeros(N_showers)

fmin_smiet = np.zeros(N_showers)
success_smiet = np.zeros(N_showers, dtype=bool)
energy_factors_smiet = np.zeros(N_showers)

fmin_smiet_vvB = np.zeros(N_showers)
success_smiet_vvB = np.zeros(N_showers, dtype=bool)
energy_factors_smiet_vvB = np.zeros(N_showers)

# for i, filename in enumerate(input_files):
#     synthesis = smietSynthesis()
#     synthesis.begin(filename, template_path=filename[:-4] + "npz")
#     # for current_sim in synthesis.origin_shower():
#     for current_sim in synthesis.run(synthesis._origin_shower):
#         Xmax_smiet[i] = current_sim.get_first_sim_shower()[shp.shower_maximum]
#
#         # filter the simulation with the same settings as before
#         efieldBandpassFilter.run(
#             current_sim,
#             current_sim.get_station().get_sim_station(),
#             None,
#             **filter_settings,
#         )
#
#         def obj(x):
#             return mean_square_error(mc_dreamland_data, current_sim, x[0])
#
#         def obj_vvB(x):
#             return mean_square_error_vvB(mc_dreamland_data, current_sim, x[0])
#
#         res = opt.minimize(obj, [1], method="Nelder-Mead")
#         res_vvB = opt.minimize(obj_vvB, [1], method="Nelder-Mead")
#
#         fmin_smiet[i] = res.fun
#         fmin_smiet_vvB[i] = res_vvB.fun
#
#         success_smiet[i] = res.success
#         success_smiet_vvB[i] = res_vvB.success
#
#         energy_factors_smiet[i] = res.x[0]
#         energy_factors_smiet_vvB[i] = res_vvB.x[0]
#
#         logger.info(
#             f"Fitting footprint{i} from file {filename} gave the following result: {res}"
#         )


# Interpolated version
interpolated_synthesis = smietInterpolated()
interpolated_synthesis.begin(input_files, template_files)

grams = (
    np.arange(
        interpolated_synthesis.synthesis[2]._origin_shower.nr_of_slices, dtype=float
    )
    + 1
)
grams *= interpolated_synthesis.synthesis[2]._origin_shower.slice_grammage

target_showers = []
target_xmax = np.array(
    list(range(650, 700, 25)) + list(range(700, 750, 5)) + list(range(750, 800, 25))
)
for xmax in target_xmax:
    target = Shower()
    target.copy_settings(interpolated_synthesis.synthesis[2]._origin_shower)
    target.long = gaisser_hillas(grams, 1e8, xmax, 210, 0.33)

    target_showers.append(target)

fmin_smiet_interpolated = np.zeros(len(target_xmax))
success_smiet_interpolated = np.zeros(len(target_xmax), dtype=bool)
energy_factors_smiet_interpolated = np.zeros(len(target_xmax))

for i, synthesised_event in enumerate(interpolated_synthesis.run(target_showers)):
    # filter the simulation with the same settings as before
    efieldBandpassFilter.run(
        synthesised_event,
        synthesised_event.get_station().get_sim_station(),
        None,
        **filter_settings,
    )

    def obj_vvB(x):
        return mean_square_error_vvB(mc_dreamland_data, synthesised_event, x[0])

    res_vvB = opt.minimize(obj_vvB, [1], method="Nelder-Mead")

    fmin_smiet_interpolated[i] = res_vvB.fun
    success_smiet_interpolated[i] = res_vvB.success
    energy_factors_smiet_interpolated[i] = res_vvB.x[0]

    logger.info(f"Fitting footprint{i} from gave the following result: \n {res_vvB}")


# And now, on to plotting!
fig, ax = plt.subplots(1, 1, figsize=(6, 8))

# ax.scatter(
#     target_xmax[success_smiet] / units.g * units.cm2,
#     fmin_smiet[success_smiet],
#     c=energy_factors_smiet[success_smiet],
# )
# ax.scatter(
#     target_xmax[success_smiet_vvB] / units.g * units.cm2,
#     fmin_smiet_vvB[success_smiet_vvB],
#     c=energy_factors_smiet_vvB[success_smiet_vvB],
#     marker="x",
# )
artist = ax.scatter(
    target_xmax[success_smiet_interpolated],
    fmin_smiet_interpolated[success_smiet_interpolated],
    c=energy_factors_smiet_interpolated[success_smiet_interpolated],
)
ax.vlines(mc_dreamland_xmax / units.g * units.cm2, *ax.get_ylim(), label="True Xmax")
ax.vlines(
    np.array(interpolated_synthesis.origin_xmax) / units.g * units.cm2,
    *ax.get_ylim(),
    color="r",
    label="Origin shower Xmax",
)
fig.colorbar(artist, label="Best fit amplitude correction factor")

ax.set_xlabel("Xmax [g/cm2]")
ax.set_ylabel("MSE")
ax.legend()

plt.show()
