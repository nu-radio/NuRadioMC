import argparse
import numpy as np
import datetime as dt

import NuRadioReco.framework.station
import NuRadioReco.framework.sim_station
import NuRadioReco.framework.radio_shower
import NuRadioReco.framework.event
from NuRadioReco.framework.parameters import showerParameters as shp

from radiotools import helper

from NuRadioMC.utilities import medium
from NuRadioReco.utilities import units, signal_processing, geometryUtilities as geo_utl
from NuRadioMC.utilities.earth_attenuation import get_weight

from NuRadioMC.SignalProp import analyticraytracing
from NuRadioReco.detector.RNO_G import rnog_detector
from NuRadioMC.simulation.simulation import calculate_sim_efield

from simulate import detector_simulation, rnog_flower_board_high_low_trigger_simulations

from plot_snr_trigger_curve import plot_trigger_effiency_snr_curve

import logging
logger = logging.getLogger("NuRadioMC.RNOG_snr_trigger_curve")
logger.setLevel(logging.INFO)


deep_trigger_channels = [0, 1, 2, 3]


def get_shower_direction_for_viewing_angle(vertex, receiver, viewing_angle, propagator):
    """
    Returns a random shower direction for a given vertex and receiver position and
    a viewing angle. Only returns directions that have a non-negligible weight to survive the earth.

    Only using direct ray for now.
    """

    propagator.set_start_and_end_point(vertex, receiver)
    propagator.find_solutions()

    if not propagator.has_solution():
        return None

    launch_zenith = helper.get_angle(propagator.get_launch_vector(0), np.array([0, 0, 1]))

    shower_phi0 = np.arctan2(receiver[1] - vertex[1], receiver[0] - vertex[0])

    idx = 0
    while True:
        ang = np.random.random() * 2 * np.pi
        unit_circle_direction = np.array([np.cos(ang), np.sin(ang), 1 / np.tan(viewing_angle)])
        unit_circle_direction = unit_circle_direction / np.linalg.norm(unit_circle_direction)

        shower_dir = np.dot(geo_utl.rot_y(launch_zenith), unit_circle_direction)
        shower_dir = np.dot(geo_utl.rot_z(shower_phi0), shower_dir)
        shower_dir /= np.linalg.norm(shower_dir)

        zenith = helper.get_zenith(-shower_dir)
        weight = get_weight(zenith, 1 * units.EeV, flavors=12, mode="core_mantle_crust_simple")

        if weight > 0.01:
            # Valid shower direction found, return it.
            break

        idx += 1
        if idx > 10:
            # Too many attempts to find a valid shower direction, return None -> try again with
            # different vertex position.
            return None

    # sanity check
    assert np.abs(viewing_angle - helper.get_angle(shower_dir, propagator.get_launch_vector(0))) < 0.01 * units.deg, \
        "Viewing angle does not match! "

    return shower_dir


def simulate_events(det, args):

    ice = medium.get_ice_model("greenland_simple")
    sampling_rate = 5 * units.GHz

    noise_vrms = signal_processing.calculate_vrms_from_temperature(
        args.noise_temperature * units.kelvin, bandwidth=[0, sampling_rate / 2])

    channel_vrms = np.zeros_like(deep_trigger_channels, dtype=float)
    for idx, channel_id in enumerate(deep_trigger_channels):
        resp = det.get_signal_chain_response(
            station_id=args.station_id, channel_id=channel_id, trigger=True)
        channel_vrms[idx] = signal_processing.calculate_vrms_from_temperature(
            args.noise_temperature * units.kelvin, response=resp)

    propagator = analyticraytracing.ray_tracing(
        ice, attenuation_model="GL3",
        detector=det,
        use_cpp=True,
        store_attenuation=False,
    )

    config = dict(
        sampling_rate = sampling_rate,
        speedup = {"redo_raytracing": False, "delta_C_cut": 0.698},
        signal = {"model": "Alvarez2009", "polarization": "auto"},
        # signal = {"model": "ARZ2020", "shift_for_xmax": True, "polarization": "auto"},
        seed = None,
    )

    ref_antenna_position = np.mean([
        det.get_relative_position(station_id=args.station_id, channel_id=channel_id)
            for channel_id in deep_trigger_channels], axis=0)
    print(f"Ref. antenna position: {ref_antenna_position}")

    events = []
    for edx in range(args.n_events):
        energy = 10 ** np.random.choice(np.linspace(*args.energy_range, 1000))

        while True:
            event = NuRadioReco.framework.event.Event(0, edx)
            station = NuRadioReco.framework.station.Station(args.station_id)
            sim_station = NuRadioReco.framework.sim_station.SimStation(args.station_id)
            sim_station.set_is_neutrino()  # better save than sorry

            if not args.fix_vertex_position:
                radius = np.random.uniform(10, 2000 ** 2) ** 0.5
                phi = np.random.uniform(0, 2 * np.pi)
                vertex_position = np.array([
                    radius * np.cos(phi),
                    radius * np.sin(phi),
                    np.random.uniform(-10, -2500)
                ])
            else:
                if args.vertex_position is not None:
                    vertex_position = np.array(args.vertex_position)
                else:
                    vertex_position = ref_antenna_position + np.array([-100, 0, 0])

            vertex_position += det.get_absolute_position(args.station_id)
            n_index = ice.get_index_of_refraction(vertex_position)
            cherenkov_angle = np.arccos(1. / n_index)

            shower_dir = get_shower_direction_for_viewing_angle(
                vertex_position, ref_antenna_position + det.get_absolute_position(args.station_id),
                cherenkov_angle + np.random.choice([-1, 1]) * args.view_angle * units.deg,
                propagator)

            if shower_dir is None:
                continue

            sim_shower = NuRadioReco.framework.radio_shower.RadioShower(0)
            zenith, azimuth = helper.cartesian_to_spherical(*(-shower_dir))
            sim_shower[shp.zenith] = zenith
            sim_shower[shp.azimuth] = azimuth
            sim_shower[shp.energy] = energy
            sim_shower[shp.flavor] = 12
            sim_shower[shp.interaction_type] = "nc"
            sim_shower[shp.type] = args.shower_type
            sim_shower[shp.vertex] = vertex_position

            for channel_id in deep_trigger_channels:
                sim_station_tmp = calculate_sim_efield(
                    [sim_shower], args.station_id, channel_id, det, propagator, medium=ice, config=config)

                station.add_sim_station(sim_station_tmp)

            event.set_station(station)
            event.add_sim_shower(sim_shower)
            if len(sim_station_tmp.get_electric_fields()):
                break

        detector_simulation(event, station, det, noise_vrms=noise_vrms, max_freq=sampling_rate / 2,
                            add_noise=args.noise)

        # fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
        # fig2, axs2 = plt.subplots(2, 2, sharex=True, sharey=True)
        # for idx, channel in enumerate(station.iter_channels()):
        #     axs.flatten()[idx].plot(channel.get_times(), channel.get_trace(), lw=1)
        #     axs2.flatten()[idx].plot(channel.get_frequencies(), np.abs(channel.get_frequency_spectrum()), lw=1)

        # axs2[0, 0].set_xlim(0.02, 0.8)
        # plt.show()

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

    parser.add_argument("--n_events", "--nevents", type=int, default=100, help="")
    parser.add_argument("--nruns", type=int, default=1, help="")
    parser.add_argument("--ncpus", type=int, default=12, help="")
    parser.add_argument("--ray", action="store_true")

    parser.add_argument("--shower_type", type=str, default="HAD", choices=["HAD", "EM"],
                        help="Type of shower to simulate (HAD for hadronic, EM for electromagnetic). Default is HAD.")
    parser.add_argument("--view_angle", type=float, default=1,
        help="Viewing angle in degrees relative to cherenkov angle.")
    parser.add_argument("--energy_range", type=float, nargs=2, default=(16, 19),
        help="Energy range in log10(E/eV) to sample the energy from. Default is (16, 19).")
    parser.add_argument("--noise_temperature", type=float, default=300,
        help="Temperature of the noise in Kelvin. Default is 300 K. This is used to calculate the vrms.")
    parser.add_argument("--fix_vertex_position", action="store_true",
                        help="Use a fixed vertex position instead of random sampling. If set either use value in `--vertex_position` or "
                        "a default position 100 m behind the reference antenna position (in x direction).")
    parser.add_argument("--vertex_position", type=float, nargs=3, default=None,
                        help="Vertex position in x,y,z coordinates (in m). Only relevant if `--fix_vertex_position` is used. ")

    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--store", action="store_true",
                        help="Store the simulated events to a file.")


    parser.add_argument("--no_noise", action="store_false", dest="noise", default=True)
    args = parser.parse_args()
    print("Using noise:", args.noise)

    if args.label != "":
        args.label = f"_{args.label}"

    logger.info(f"Simulate the noise trigger rate for station {args.station_id} "
                f"(using channels {deep_trigger_channels})")

    det = rnog_detector.Detector(
        detector_file=args.detectordescription, log_level=logging.INFO,
        always_query_entire_description=True, select_stations=args.station_id)

    det.update(dt.datetime(2024, 1, 3))

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

        ray.init(num_cpus=args.ncpus)
        @ray.remote
        def simulate_events_ray(*args, **kwargs):
            return simulate_events(*args, **kwargs)

        det_ref = ray.put(det)
        args_ref = ray.put(args)

        remotes = [simulate_events_ray.remote(det_ref, args_ref) for _ in range(args.nruns)]
        events = np.hstack(ray.get(remotes))

    else:
        events = simulate_events(det, args)
        events = np.array(events)

    print(f"Simulated {len(events)} events in total.")

    if args.label != "":
        args.label = f"_{args.label}"

    if args.store:
        from NuRadioReco.modules.io.eventWriter import eventWriter as EventWriter
        writer = EventWriter()
        writer.begin(f"rnog_snr_trigger_curve_events_viewing_angle-{args.view_angle}deg"
                     f"_{10 ** args.energy_range[0]:.2e}-{10 ** args.energy_range[1]:.2e}eV_station-{args.station_id}"
                     f"_{args.shower_type}{'_random_vertices' if not args.fix_vertex_position else ''}{args.label}")

        for event in events:
            writer.run(event, mode={
                'Channels': True,
                'ElectricFields': False,
                'SimChannels': False,
                'SimElectricFields': False
            })
        writer.end()

    # plot_trigger_effiency_snr_curve(events, None)