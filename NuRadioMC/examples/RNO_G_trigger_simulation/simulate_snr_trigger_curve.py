import argparse
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from collections import defaultdict

import NuRadioReco.framework.electric_field
import NuRadioReco.framework.station
import NuRadioReco.framework.sim_station
import NuRadioReco.framework.event
from NuRadioReco.framework.parameters import electricFieldParameters as efp

from radiotools import helper

from NuRadioMC.utilities import medium
from NuRadioReco.utilities import units, fft, signal_processing, trace_utilities
from NuRadioMC.SignalGen import askaryan
from NuRadioMC.SignalProp import analyticraytracing
from NuRadioReco.detector.RNO_G import rnog_detector
from NuRadioMC.simulation.simulation import calculate_polarization_vector

from simulate import detector_simulation, rnog_flower_board_high_low_trigger_simulations

import logging
logger = logging.getLogger("NuRadioMC.RNOG_snr_trigger_curve")
logger.setLevel(logging.INFO)


def binomial_proportion(nsel, ntot, coverage=0.68):
    """
    Copied from pyik.mumpyext (original author HD)

    Calculate a binomial proportion (e.g. efficiency of a selection) and its confidence interval.

    Parameters
    ----------
    nsel: array-like
        Number of selected events.
    ntot: array-like
        Total number of events.
    coverage: float (optional)
        Requested fractional coverage of interval (default: 0.68).

    Returns
    -------
    p: array of dtype float
        Binomial fraction.
    dpl: array of dtype float
        Lower uncertainty delta (p - pLow).
    dpu: array of dtype float
        Upper uncertainty delta (pUp - p).

    Examples
    --------
    >>> p, dpl, dpu = binomial_proportion(50, 100, 0.68)
    >>> print(f"{p:.4f} {dpl:.4f} {dpu:.4f}"
    0.5000 0.0495 0.0495
    >>> abs(np.sqrt(0.5*(1.0-0.5)/100.0)-0.5*(dpl+dpu)) < 1e-3
    True

    Notes
    -----
    The confidence interval is approximate and uses the score method
    of Wilson. It is based on the log-likelihood profile and can
    undercover the true interval, but the coverage is on average
    closer to the nominal coverage than the exact Clopper-Pearson
    interval. It is impossible to achieve perfect nominal coverage
    as a consequence of the discreteness of the data.
    """

    from scipy.stats import norm

    z = norm().ppf(0.5 + 0.5 * coverage)
    z2 = z * z
    p = np.asarray(nsel, dtype=float) / ntot
    div = 1.0 + z2 / ntot
    pm = (p + z2 / (2 * ntot))
    dp = z * np.sqrt(p * (1.0 - p) / ntot + z2 / (4 * ntot * ntot))
    pl = (pm - dp) / div
    pu = (pm + dp) / div

    return p, p - pl, pu - p


deep_trigger_channels = [0, 1, 2, 3]


def get_propagation_result(det, vertex_position):
    ice = medium.get_ice_model("greenland_simple")

    propagator = analyticraytracing.ray_tracing(
        ice, attenuation_model="GL3",
        detector=det,
        use_cpp=True,
    )

    results = defaultdict(dict)

    for channel_id in deep_trigger_channels:

        antenna_position = det.get_relative_position(station_id=args.station_id, channel_id=channel_id)
        propagator.set_start_and_end_point(vertex_position, antenna_position)
        propagator.find_solutions()

        results["distance"][channel_id] = propagator.get_path_length(0)
        results["travel_time"][channel_id] = propagator.get_travel_time(0)

        results["launch_vector"][channel_id] = propagator.get_launch_vector(0)
        results["receive_vector"][channel_id] = propagator.get_receive_vector(0)

        results["launch_zenith"][channel_id] = helper.get_angle(results["launch_vector"][channel_id], np.array([1, 0, 0]))

    return results


def simulate_events(det, viewing_angle_c, vertex_position, e_min=15.5, e_max=16.3, n_events=1000):

    shower_type = "HAD"
    n_index = 1.78
    cherenkov_angle = np.arccos(1. / n_index)
    n_samples = 1024
    sampling_rate = 5 * units.GHz
    config = {"signal": {"polarization": "auto"}}

    noise_vrms = signal_processing.calculate_vrms_from_temperature(300 * units.kelvin, bandwidth=[0, sampling_rate / 2])

    channel_vrms = []
    for channel_id in deep_trigger_channels:
        resp = det.get_signal_chain_response(station_id=args.station_id, channel_id=channel_id, trigger=True)
        channel_vrms.append(signal_processing.calculate_vrms_from_temperature(300 * units.kelvin, response=resp))
    channel_vrms = np.array(channel_vrms)

    propagation_results = get_propagation_result(det, vertex_position)

    events = []
    snr_events = []
    for _ in range(n_events):
        energy = 10 ** np.random.choice(np.linspace(e_min, e_max, 1000))

        event = NuRadioReco.framework.event.Event(0, 0)
        station = NuRadioReco.framework.station.Station(args.station_id)
        sim_station = NuRadioReco.framework.sim_station.SimStation(args.station_id)

        for channel_id in deep_trigger_channels:

            viewing_angle = cherenkov_angle + viewing_angle_c * units.deg
            shower_zenith = np.pi / 2 + viewing_angle + propagation_results["launch_zenith"][channel_id]
            shower_direction = helper.spherical_to_cartesian(zenith=shower_zenith, azimuth=0)

            spectrum = askaryan.get_frequency_spectrum(
                energy, viewing_angle,
                n_samples, 1 / sampling_rate, shower_type, n_index, propagation_results["distance"][channel_id],
                "ARZ2020", seed=None, full_output=False, shift_for_xmax=True,
                same_shower=channel_id != deep_trigger_channels[0])

            polarization_direction_onsky = calculate_polarization_vector(
                shower_direction, propagation_results["launch_vector"][channel_id], config)

            # eR, eTheta, ePhi
            spectra = np.outer(polarization_direction_onsky, spectrum)

            # this is common stuff which is the same between emitters and showers
            electric_field = NuRadioReco.framework.electric_field.ElectricField([channel_id],
                                    position=det.get_relative_position(args.station_id, channel_id),
                                    shower_id=0, ray_tracing_id=0)
            electric_field.set_frequency_spectrum(spectra, sampling_rate)
            electric_field.set_trace_start_time(propagation_results["travel_time"][channel_id])

            electric_field[efp.azimuth] = 0
            electric_field[efp.zenith] = helper.get_zenith(propagation_results["launch_vector"][channel_id])
            electric_field[efp.ray_path_type] = 0

            sim_station.add_electric_field(electric_field)
            sim_station.set_is_neutrino()  # better save than sorry
            station.set_sim_station(sim_station)

        event.set_station(station)
        detector_simulation(event, station, det, noise_vrms=noise_vrms, max_freq=sampling_rate / 2)

        rnog_flower_board_high_low_trigger_simulations(
            event, station, det, deep_trigger_channels, channel_vrms, {"sigma_3.7": 3.7}, {"sigma_3.0": 3.0})

        events.append(event)

    return events



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run NuRadioMC simulation")
    # Sim steering arguments
    parser.add_argument("--detectordescription", '--det', type=str, default=None,
        help="Path to RNO-G detector description file. If None, query from DB")
    parser.add_argument("--station_id", type=int, default=None,
        help="Set station to be used for simulation", required=True)
    parser.add_argument("--nevents", type=int, default=200, help="")
    parser.add_argument("--nruns", type=int, default=1, help="")
    parser.add_argument("--ncpus", type=int, default=12, help="")
    parser.add_argument("--n_samples", type=int, default=1024, help="")
    parser.add_argument("--index", type=int, default=0, help="")
    parser.add_argument("--ray", action="store_true")
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--noise_type", type=str, default="rayleigh")
    parser.add_argument("--running_vrms", action="store_true")
    parser.add_argument("--noise_temperature", type=float, default=300,
        help="Temperature of the noise in Kelvin. Default is 300 K. This is used to calculate the vrms.")
    parser.add_argument("--correlated_noise_temperature", type=float, default=None,
        help="Temperature of the correlated noise in Kelvin. Default is None. This is used to calculate the vrms.")

    args = parser.parse_args()

    if args.noise_type != "rayleigh":
        raise NotImplementedError("Only \"rayleigh\" noise is currently implemented.")

    if args.label != "":
        args.label = f"_{args.label}"

    logger.info(f"Simulate the noise trigger rate for station {args.station_id} "
                f"(using channels {deep_trigger_channels})")

    det = rnog_detector.Detector(
        detector_file=args.detectordescription, log_level=logging.INFO,
        always_query_entire_description=True, select_stations=args.station_id)

    det.update(dt.datetime(2024, 1, 3))

    antenna_position = np.mean([
        det.get_relative_position(station_id=args.station_id, channel_id=channel_id)
        for channel_id in deep_trigger_channels], axis=0)

    # antenna_position = det.get_relative_position(station_id=args.station_id, channel_id=2)
    print(f"Antenna position: {antenna_position}")

    vertex_position = antenna_position + np.array([-100, 0, 0])

    if args.ray:
        """
        Ray is a powerful library for parallel computing. It allows to easily parallelize the simulation
        on multiple CPUs. Do no use this in combination with batch system (like slurm or HTConder).
        But if you are on a machine with many cores you can run this script interactively.
        """
        try:
            import ray
        except ImportError:
            raise ImportError("You passed the option '--ray' but ray is not installed. "
                              "Either install ray with 'pip install ray' or remove the commandline flag.")

        # antenna_pattern_provider is a singleton - load antenna patterns upfront
        import NuRadioReco.detector.antennapattern
        antenna_pattern_provider = NuRadioReco.detector.antennapattern.AntennaPatternProvider()
        for channel_id in deep_trigger_channels:
            antenna_pattern_provider.load_antenna_pattern(
                det.get_antenna_model(args.station_id, channel_id), do_consistency_check=False)

        ray.init(num_cpus=args.ncpus)
        @ray.remote
        def simulate_events_ray(*args, **kwargs):
            return simulate_events(*args, **kwargs)

        det_ref = ray.put(det)
        vertex_position_ref = ray.put(vertex_position)
        n_events_ref = ray.put(args.nevents)

        remotes = [simulate_events_ray.remote(det_ref, 1, vertex_position, n_events=n_events_ref)
            for _ in range(args.nruns)]
        events = np.hstack(ray.get(remotes))
        print(f"Simulated {len(events)} events in total.")

    else:
        events = simulate_events(det, 4, vertex_position, n_events=args.nevents)


    snr_events = []
    for event in events:
        station = event.get_station()
        snrs_event = []
        for channel in station.iter_trigger_channels():
            trace = channel.get_trace()
            snr = trace_utilities.get_signal_to_noise_ratio(
                trace, trace_utilities.get_split_trace_noise_RMS(trace))
            snrs_event.append(snr)

        snr_events.append(np.sort(snrs_event)[-2])

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    n, bins = np.histogram(snr_events, bins=np.linspace(2, 6, 40))

    trigger_fractions = []
    for idx in range(len(n)):
        bin_mask = (snr_events >= bins[idx]) & (snr_events < bins[idx + 1])
        sel_events = events[bin_mask]
        coinc_p, coinc_p_low, coinc_p_up = binomial_proportion(
            np.sum([event.has_triggered("deep_high_low_sigma_3.7") for event in sel_events]), len(sel_events))
        pa_p, pa_p_low, pa_p_up = binomial_proportion(
            np.sum([event.has_triggered("pa_power_sigma_3.0") for event in sel_events]), len(sel_events))

        trigger_fractions.append(
            np.around([coinc_p, coinc_p_low, coinc_p_up,
            pa_p, pa_p_low, pa_p_up], 5))

    trigger_fractions = np.array(trigger_fractions)
    ax.errorbar(bins[:-1] + np.diff(bins) / 2, trigger_fractions[:, 0],
        yerr=[trigger_fractions[:, 1], trigger_fractions[:, 2]], color="C0", ls=":", lw=1, marker="o")
    ax.errorbar(bins[:-1] + np.diff(bins) / 2, trigger_fractions[:, 3],
        yerr=[trigger_fractions[:, 4], trigger_fractions[:, 5]], color="C1", ls=":", lw=1, marker="s")

    ax.plot(np.nan, np.nan, color="C0", marker="o", label="high-low trigger")
    ax.plot(np.nan, np.nan, color="C1", marker="s", label="phased-array trigger")
    ax.set_xlim(None, 6)
    ax.set_xlabel("SNR")
    ax.set_ylabel("Trigger fraction")
    ax.grid()
    ax.legend()
    fig.tight_layout()
    plt.show()