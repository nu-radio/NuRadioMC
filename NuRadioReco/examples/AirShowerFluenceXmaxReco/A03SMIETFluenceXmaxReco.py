import os
import glob
import h5py
import logging
import argparse
import numpy as np

from matplotlib import pyplot as plt
from scipy import optimize as opt
from scipy import constants
from astropy import time

from NuRadioReco.framework import station
import NuRadioReco.framework.event
import NuRadioReco.modules.electricFieldBandPassFilter
import NuRadioReco.modules.electricFieldSignalReconstructor
import NuRadioReco.modules.efieldGalacticNoiseAdder
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.eventTypeIdentifier
import NuRadioReco.modules.channelSignalReconstructor
import NuRadioReco.modules.electricFieldSignalReconstructor
import NuRadioReco.modules.electricFieldBandPassFilter
import NuRadioReco.modules.voltageToEfieldConverterPerChannelGroup

from NuRadioReco.utilities import units
from NuRadioReco.detector import detector
from NuRadioReco.modules.io.coreas import coreas, coreasInterpolator, readCoREASDetector
from NuRadioReco.modules.template_synthesis.smietSynthesis import smietInterpolated
from NuRadioReco.utilities.dataservers import download_from_dataserver
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import electricFieldParameters as efp

from smiet.numpy import Shower, CoreasHDF5


# Create logger
logger = logging.getLogger("NuRadioReco.SMIETFluenceXmaxReco")


# Global variables
remote_path = "data/CoREAS/LOFAR/evt_00001"
path = "data/CoREAS/LOFAR/evt_00001"

input_files = glob.glob(os.path.join(path, "*.hdf5"))
N_showers = len(input_files)

origin_files = [
    "/home/mitjadesmet/Data/Showers_for_Xmax_reco/vertical_geometry/SIM900204.hdf5",
    "/home/mitjadesmet/Data/Showers_for_Xmax_reco/vertical_geometry/SIM900100.hdf5",
    "/home/mitjadesmet/Data/Showers_for_Xmax_reco/vertical_geometry/SIM900101.hdf5",
]

# Initialize detector
detector = detector.Detector(
    json_filename="grid_array_SKALA_InfFirn.json",
    assume_inf=False,
    antenna_by_depth=False,
)
detector.update(time.Time("2023-01-01T00:00:00", format="isot", scale="utc"))

# Initialize the modules
efieldBandpassFilter = (
    NuRadioReco.modules.electricFieldBandPassFilter.electricFieldBandPassFilter()
)
electricFieldSignalReconstructor = NuRadioReco.modules.electricFieldSignalReconstructor.electricFieldSignalReconstructor()
efieldGalacticNoiseAdder = (
    NuRadioReco.modules.efieldGalacticNoiseAdder.efieldGalacticNoiseAdder()
)
efieldToVoltageConverter = (
    NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter(
        log_level=logging.INFO
    )
)
efieldToVoltageConverter.begin(debug=False, pre_pulse_time=0, post_pulse_time=0)
channelGenericNoiseAdder = (
    NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
)
triggerSimulator = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()
triggerSimulator.begin()
channelBandPassFilter = (
    NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
)
channelBandPassFilter.begin()
eventTypeIdentifier = NuRadioReco.modules.eventTypeIdentifier.eventTypeIdentifier()
channelSignalReconstructor = (
    NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
)
channelSignalReconstructor.begin()
voltageToEfieldConverterPerChannelGroup = NuRadioReco.modules.voltageToEfieldConverterPerChannelGroup.voltageToEfieldConverterPerChannelGroup()
voltageToEfieldConverterPerChannelGroup.begin(use_MC_direction=True)

coreas_reader = readCoREASDetector.readCoREASDetector()

# Filter settings
filter_settings = {
    "passband": [30 * units.MHz, 80 * units.MHz],
    "filter_type": "butter",
    "order": 10,
}


# Define some functions for later
def calculate_fluence_around_peak(
    trace,
    sampling,
    signal_window: float = 25 * units.ns,
    sample_axis: int = 1,
    return_uncertainty=False,
):
    conversion_factor_integrated_signal = (
        constants.c * constants.epsilon_0 * units.joule / units.s / units.volt**2
    )

    peak_sample = np.argmax(trace, axis=sample_axis)
    window_lower = np.clip(
        peak_sample - int(signal_window / sampling), 0, trace.shape[sample_axis]
    )
    window_higher = np.clip(
        peak_sample + int(signal_window / sampling), 0, trace.shape[sample_axis]
    )

    fluence = []
    fluence_error = []
    for signal, low, high in zip(trace, window_lower, window_higher):
        noise_start = -500

        f_signal = np.sum(signal[low:high] ** 2)
        f_noise = np.sum(signal[noise_start:] ** 2)

        f_signal -= f_noise * (high - low) / len(signal[noise_start:])
        if f_signal < 0:
            f_signal = 0

        RMSNoise = np.sqrt(np.mean(signal[noise_start:] ** 2))

        signal_energy_fluence = (
            f_signal * sampling * conversion_factor_integrated_signal
        )

        signal_window_duration = (high - low) * sampling
        signal_energy_fluence_error = (
            4
            * np.abs(signal_energy_fluence / conversion_factor_integrated_signal)
            * RMSNoise**2
            * sampling
            + 2 * signal_window_duration * RMSNoise**4 * sampling
        ) ** 0.5 * conversion_factor_integrated_signal

        fluence.append(np.sum(signal_energy_fluence))
        fluence_error.append(np.sum(signal_energy_fluence_error))

    if return_uncertainty:
        return np.asarray(fluence), np.asarray(fluence_error)

    return np.asarray(fluence)


def mean_square_error(
    evt_data: NuRadioReco.framework.event.Event,
    evt_sim: NuRadioReco.framework.event.Event,
):
    data_fluences = []
    data_positions = []
    for efield in evt_data.get_station().get_sim_station().get_electric_fields():
        data_fluences.append(
            np.sum(
                calculate_fluence_around_peak(
                    efield.get_trace()[1:], efield.get_sampling_rate()
                )
            )
        )
        data_positions.append(efield.get_position())
    data_fluences = np.asarray(data_fluences)
    data_positions = np.asarray(data_positions)

    sim_fluences = []
    sim_positions = []
    for efield in evt_sim.get_station().get_sim_station().get_electric_fields():
        sim_fluences.append(
            np.sum(
                calculate_fluence_around_peak(
                    efield.get_trace()[1:], efield.get_sampling_rate()
                )
            )
        )
        sim_positions.append(np.squeeze(efield.get_position()))
    sim_fluences = np.asarray(sim_fluences)
    sim_positions = np.asarray(sim_positions)

    # Sort the arrays by position, to ensure matching same antennas
    data_ind = np.lexsort((data_positions[:, 1], data_positions[:, 0]))
    sim_ind = np.lexsort((sim_positions[:, 1], sim_positions[:, 0]))

    return (
        data_fluences[data_ind],
        sim_fluences[sim_ind],
        data_positions[data_ind],
        sim_positions[sim_ind],
    )


def get_fluences(
    evt: NuRadioReco.framework.event.Event,
):
    fluences = []
    errors = []
    positions = []
    for efield in evt.get_station().get_sim_station().get_electric_fields():
        f, f_error = calculate_fluence_around_peak(
            efield.get_trace(), efield.get_sampling_rate(), return_uncertainty=True
        )
        fluences.append(np.sum(f))
        errors.append(np.sum(f_error))
        positions.append(efield.get_position())
    fluences = np.asarray(fluences)
    errors = np.asarray(errors)
    positions = np.asarray(positions)

    # Sort the arrays by position, to ensure matching same antennas
    ind = np.lexsort((positions[:, 1], positions[:, 0]))

    return (
        fluences[ind],
        errors[ind],
        positions[ind],
    )


def gaisser_hillas(X, Nmax, Xmax, L, R):
    Xprime = X - Xmax
    t1 = 1 + R * Xprime / L
    pow = R**-2
    exp = -Xprime / (L * R)

    N = Nmax * t1**pow * np.exp(exp)
    N = np.nan_to_num(N)
    N = np.where(N < 1, 1, N)

    return np.stack((X, N), axis=1)


def obj(x, data_event, sim_event):
    data, sim, data_pos, sim_pos = mean_square_error(data_event, sim_event)

    data_ant_at_core = np.linalg.norm(data_pos, axis=1) < 1 * units.m
    sim_ant_at_core = np.linalg.norm(sim_pos, axis=1) < 1 * units.m

    return np.sum((data[~data_ant_at_core] - x[0] * sim[~sim_ant_at_core]) ** 2)


def obj_core_fit(x, data_event, interp):
    core = np.asarray([x[0], x[1], 0]) * units.m

    data, error, data_pos = get_fluences(data_event)

    data_ant_at_core = np.linalg.norm(data_pos, axis=1) < 1 * units.m

    sim = np.zeros_like(data[~data_ant_at_core])
    for ind, pos in enumerate(data_pos[~data_ant_at_core]):
        sim[ind] = interp.get_interp_fluence_value(pos - core)

    return np.sum(
        ((data[~data_ant_at_core] - x[2] * sim) / (2 * error[~data_ant_at_core])) ** 2
    )


def perform_smiet_fit(
    the_synthesis, grams, target_xmax, target_nmax, target_L, target_R
):
    target_showers = []
    for xmax in target_xmax:
        target = Shower()
        target.copy_settings(the_synthesis.synthesis[2].get_origin_shower())
        target.long = gaisser_hillas(grams, target_nmax, xmax, target_L, target_R)

        target_showers.append(target)

    fmin = np.zeros(len(target_showers))
    success = np.zeros(len(target_showers), dtype=bool)
    energy_factors = np.zeros((len(target_showers), 3))

    all_synth_events = []
    for i, synthesised_event in enumerate(the_synthesis.run(target_showers)):
        # filter the simulation with the same settings as before
        efieldBandpassFilter.run(
            synthesised_event,
            synthesised_event.get_station().get_sim_station(),
            None,
            **filter_settings,
        )

        # Initialize interpolator for core position fit
        interpolator = coreasInterpolator.coreasInterpolator(synthesised_event)
        interpolator.set_fluence_of_efields(
            lambda trace: calculate_fluence_around_peak(
                trace[1:],
                synthesised_event.get_station()
                .get_sim_station()
                .get_electric_fields()[0]
                .get_sampling_rate(),
            )
        )
        interpolator.initialize_fluence_interpolator()

        res = opt.minimize(
            lambda x: obj_core_fit(x, mc_dreamland_data, interpolator),
            [0, 0, 1],
            method="Nelder-Mead",
        )

        all_synth_events.append(synthesised_event)

        fmin[i] = res.fun
        success[i] = res.success
        energy_factors[i] = res.x

        logger.info(f"Fitting footprint{i} from gave the following result: \n {res}")

    return all_synth_events, fmin, success, energy_factors


def read_data_event(in_file, det, filt_settings=filter_settings):
    mc_dreamland_data = coreas.read_CORSIKA7(in_file)
    efieldBandpassFilter.run(
        mc_dreamland_data,
        mc_dreamland_data.get_station().get_sim_station(),
        det,
        **filter_settings,
    )

    mc_dreamland_data.get_station().get_sim_station().set_station_time(
        time.Time("2023-01-01T00:00:00", format="isot", scale="utc")
    )
    mc_dreamland_data.get_station().get_sim_station()._station_id = (
        1  # HACK to work with json detector
    )
    efieldGalacticNoiseAdder.begin()
    efieldGalacticNoiseAdder.run(
        mc_dreamland_data,
        mc_dreamland_data.get_station().get_sim_station(),
        det,
        passband=filt_settings["passband"],
    )

    return mc_dreamland_data


def read_data_event_with_noise(
    in_file, det, filt_settings=filter_settings, core=np.array([0, 0, 0]) * units.m
):
    from scipy import constants

    coreas_reader.begin(in_file)
    # we only need a single realization of the shower, so we set the core position to zero for simplicity
    for _, evt in enumerate(coreas_reader.run(det, [core])):
        station = evt.get_station()

    # apply antenna response
    efieldToVoltageConverter.run(evt, station, det)

    # approximate the rest of the signal chain with a bandpass filter
    channelBandPassFilter.run(evt, station, det, **filt_settings)

    # add thermal noise of fixed noise temperature
    Tnoise = 300  # in Kelvin

    # calculate Vrms and normalize such that after filtering the correct Vrms is obtained
    min_freq = 0
    max_freq = 0.5 * det.get_sampling_frequency(station.get_id(), 1)
    ff = np.linspace(0, max_freq, 10000)
    filt = channelBandPassFilter.get_filter(
        ff, station.get_id(), None, det, **filt_settings
    )
    bandwidth = np.trapz(np.abs(filt) ** 2, ff)
    Vrms = (Tnoise * 50 * constants.k * bandwidth / units.Hz) ** 0.5
    amplitude = Vrms / (bandwidth / max_freq) ** 0.5
    channelGenericNoiseAdder.run(
        evt,
        station,
        det,
        type="rayleigh",
        amplitude=1 * amplitude,
        min_freq=min_freq,
        max_freq=max_freq,
    )

    # simulate if the air shower triggers the station. Requireing a 10 sigma simple threshold trigger
    triggerSimulator.run(evt, station, det, number_concidences=1, threshold=10 * Vrms)
    if station.get_trigger("default_simple_threshold").has_triggered():
        eventTypeIdentifier.run(evt, station, "forced", "cosmic_ray")
        channelSignalReconstructor.run(evt, station, det)

        # reconstruct the electric field for each dual-polarized antenna through standard unfolding
        voltageToEfieldConverterPerChannelGroup.run(evt, station, det)

        # calculate the electric field parameters
        station = evt.get_station().get_sim_station()
        electricFieldSignalReconstructor.run(evt, station, det)

        return evt

    else:
        logger.error("The station has not triggered")
        raise RuntimeError("The station has not triggered")


def get_efield_properties(evt):
    station = evt.get_station().get_sim_station()

    ff = []
    ff_error = []
    pos = []
    for efield in station.get_electric_fields():
        ff.append(np.sum(efield[efp.signal_energy_fluence]))
        ff_error.append(np.sum(efield.get_parameter_error(efp.signal_energy_fluence)))
        pos.append(efield.get_position())

    pos = np.array(pos)
    ff = np.array(ff)
    ff_error = np.array(ff_error)
    ff_error[ff_error == 0] = ff[
        ff_error != 0
    ].min()  # ensure that the error is always larger than 0 to avoid division by zero in objective function

    return ff, ff_error, pos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-event", type=int, default=10)
    parser.add_argument("--logging", type=int, default=20)
    parser.add_argument("--synth-freq", nargs=2, type=int, default=[30, 500])

    args = parser.parse_args()

    # Set logger level
    logger.setLevel(args.logging)

    # Create directory for plots
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # Download coreas simulations, if they are not present in the directory
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

    synth_freq = args.synth_freq

    template_files = [
        f"/home/mitjadesmet/Data/Showers_for_Xmax_reco/vertical_geometry/templates_{synth_freq[0]}_{synth_freq[1]}/"
        + os.path.basename(file)[:-4]
        + "npz"
        for file in origin_files
    ]

    # Read in the "data" shower and find the profile variables
    input_file = input_files[args.data_event]
    mc_dreamland_shower = CoreasHDF5(input_file)

    long = mc_dreamland_shower.get_long_profile()
    popt, _ = opt.curve_fit(
        lambda x, a, b, c, d: gaisser_hillas(x, a, b, c, d)[:, 1],
        long[:, 0],
        np.sum(long[:, 2:4], axis=1),
        p0=[1e8, 700, 200, 0.3],
    )
    mc_dreamland_nmax, mc_dreamland_xmax, mc_dreamland_L, mc_dreamland_R = popt

    # Check if xmax is within the range for SMIET interpolation
    if mc_dreamland_xmax > 823 or mc_dreamland_xmax < 624:
        logger.error(
            f"The data Xmax {mc_dreamland_xmax} is outside of interpolation range"
        )
        raise RuntimeError

    # Read in the "data" Event, filter the electric fields and add noise
    mc_dreamland_data = read_data_event(input_file, detector)

    # STEP 1: fit to CoREAS simulations
    # Initialize arrays to store the fit results
    fmin = np.zeros(N_showers)
    success = np.zeros(N_showers, dtype=bool)
    Xmax = np.zeros(N_showers)
    energy_factors = np.zeros((N_showers, 3))

    # Loop over all CoREAS simulations and compute chi2
    for i, filename in enumerate(input_files):
        # skip the CoREAS simulation that was used to generate the event under test
        if filename == input_file:
            continue

        # read in CoREAS simulation
        current_sim: NuRadioReco.framework.event.Event = coreas.read_CORSIKA7(filename)
        Xmax[i] = current_sim.get_first_sim_shower()[shp.shower_maximum]

        # filter the simulation with the same settings as before
        efieldBandpassFilter.run(
            current_sim,
            current_sim.get_station().get_sim_station(),
            None,
            **filter_settings,
        )
        # electricFieldSignalReconstructor.run(current_sim,current_sim.get_station().get_sim_station(), None)

        # Initialize interpolator for core position fit
        interpolator = coreasInterpolator.coreasInterpolator(current_sim)
        interpolator.set_fluence_of_efields(
            lambda trace: calculate_fluence_around_peak(
                trace[1:],
                current_sim.get_station()
                .get_sim_station()
                .get_electric_fields()[0]
                .get_sampling_rate(),
            )
        )
        interpolator.initialize_fluence_interpolator()

        res = opt.minimize(
            lambda x: obj_core_fit(x, mc_dreamland_data, interpolator),
            [0, 0, 1],
            method="Nelder-Mead",
        )

        fmin[i] = res.fun
        success[i] = res.success
        energy_factors[i] = res.x
        logger.info(
            f"Fitting footprint{i} from file {filename} gave the following result: {res}"
        )

    # STEP 2: Interpolated SMIET fitting
    interpolated_synthesis = smietInterpolated(freq=synth_freq)
    interpolated_synthesis.begin(origin_files, template_files)

    # Optionally save the templates
    if 0:
        for synthesis in interpolated_synthesis.synthesis:
            synthesis.save_template(
                save_dir=f"/home/mitjadesmet/Data/Showers_for_Xmax_reco/vertical_geometry/templates_{synth_freq[0]}_{synth_freq[1]}/"
            )

    grams = np.asarray(interpolated_synthesis.synthesis[2].slices_grammage)
    target_xmax_broad = np.arange(625, 825, 25)

    (
        _,
        fmin_smiet_interpolated_broad,
        success_smiet_interpolated_broad,
        energy_factors_smiet_interpolated_broad,
    ) = perform_smiet_fit(
        interpolated_synthesis,
        grams,
        target_xmax_broad,
        mc_dreamland_nmax,
        mc_dreamland_L,
        mc_dreamland_R,
    )

    min_index = np.argsort(fmin_smiet_interpolated_broad)[:2]
    lower_xmax = min(target_xmax_broad[min_index])
    upper_xmax = max(target_xmax_broad[min_index])
    target_xmax = np.arange(lower_xmax, upper_xmax, 3)
    # target_xmax = np.arange(700, 750, 3)

    (
        _,
        fmin_smiet_interpolated,
        success_smiet_interpolated,
        energy_factors_smiet_interpolated,
    ) = perform_smiet_fit(
        interpolated_synthesis,
        grams,
        target_xmax,
        mc_dreamland_nmax,
        mc_dreamland_L,
        mc_dreamland_R,
    )

    target_xmax = np.concat((target_xmax_broad, target_xmax))
    fmin_smiet_interpolated = np.concat(
        (fmin_smiet_interpolated_broad, fmin_smiet_interpolated)
    )
    energy_factors_smiet_interpolated = np.concat(
        (energy_factors_smiet_interpolated_broad, energy_factors_smiet_interpolated)
    )
    success_smiet_interpolated = np.concat(
        (success_smiet_interpolated_broad, success_smiet_interpolated)
    )

    # Save the results for analysis
    with h5py.File(f"data/SMIET/{os.path.basename(input_file)}", "w") as file:
        coreas_group = file.create_group("CoREAS")

        coreas_group.create_dataset("fmin", data=fmin)
        coreas_group.create_dataset("Xmax", data=Xmax / units.g * units.cm2)
        coreas_group.create_dataset("success", data=success)
        coreas_group.create_dataset("energy_factors", data=energy_factors)

        smiet_group = file.create_group("SMIET")

        smiet_group.create_dataset("fmin", data=fmin_smiet_interpolated)
        smiet_group.create_dataset("Xmax", data=target_xmax)
        smiet_group.create_dataset("success", data=success_smiet_interpolated)
        smiet_group.create_dataset(
            "energy_factors", data=energy_factors_smiet_interpolated
        )

        metadata = file.create_group("metadata")

        metadata.create_dataset(
            "mc_dreamland",
            data=[mc_dreamland_nmax, mc_dreamland_xmax, mc_dreamland_L, mc_dreamland_R],
        )
        metadata.create_dataset(
            "interpolated_synthesis",
            data=np.array(interpolated_synthesis.origin_xmax) / units.g * units.cm2,
        )

    # And now, on to plotting!
    marker = ["x", "o"]

    fig, ax = plt.subplots(1, 1, figsize=(6, 8))

    ax.scatter(
        Xmax[success] / units.g * units.cm2,
        fmin[success],
        c=energy_factors[success][:, 2],
        marker=marker[0],
        label="Fit to CoREAS showers",
    )

    artist = ax.scatter(
        target_xmax[success_smiet_interpolated],
        fmin_smiet_interpolated[success_smiet_interpolated],
        c=energy_factors_smiet_interpolated[success_smiet_interpolated][:, 2],
        marker=marker[1],
        label="Fit to interpolated SMIET synthesis",
    )
    ax.vlines(mc_dreamland_xmax, *ax.get_ylim(), label="True Xmax")
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

    fig.savefig("plots/xmax_fit_SMIET_coreas.png", dpi=250, bbox_inches="tight")

    # plt.show()
