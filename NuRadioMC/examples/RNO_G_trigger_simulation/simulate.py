#!/bin/env python3

import argparse
import numpy as np
import os
import secrets
import functools
import datetime as dt

from NuRadioMC.EvtGen import generator
from NuRadioMC.simulation import simulation
from NuRadioReco.utilities import units, signal_processing, fft

from NuRadioReco.detector.RNO_G import rnog_detector
from NuRadioReco.detector.response import Response

import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.channelGenericNoiseAdder

from NuRadioReco.modules.RNO_G import hardwareResponseIncorporator, triggerBoardResponse
from NuRadioReco.modules.trigger import highLowThreshold

import logging
logger = logging.getLogger("NuRadioMC.RNOG_trigger_simulation")
logger.setLevel(logging.INFO)


deep_trigger_channels = np.array([0, 1, 2, 3])

efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin(caching=False, pre_pulse_time=400 * units.ns)

channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
channelGenericNoiseAdder.begin()

rnogHardwareResponse = hardwareResponseIncorporator.hardwareResponseIncorporator()
rnogHardwareResponse.begin(trigger_channels=deep_trigger_channels)

highLowThreshold = highLowThreshold.triggerSimulator()
rnogADCResponse = triggerBoardResponse.triggerBoardResponse()
rnogADCResponse.begin(clock_offset=0, adc_output="counts")


def detector_simulation(evt, station, det, noise_vrms, max_freq, add_noise=True):
    """ Run the detector simulation.

    It performs the following steps:
    - efieldToVoltageConverter: Convert the electric fields to voltages
    - channelGenericNoiseAdder: Add noise to the channels
    - rnogHardwareResponse: Apply the hardware response (for RADIANT and FLOWER channels)

    Parameters
    ----------
    evt : NuRadioMC.framework.event.Event
        The event to simulate the detector response for.
    station : NuRadioMC.framework.station.Station
        The station to simulate the detector response for.
    det : NuRadioReco.detector.RNO_G.rnog_detector.Detector
        The detector description.
    noise_vrms : float or dict of list
        The noise vrms (without any filter!). If a dict is given, the keys are the channel ids.
    max_freq : float
        The maximum frequency for the noise, i.e., the nyquist frequency for the simulated sampling rate.
    """

    efieldToVoltageConverter.run(evt, station, det, channel_ids=deep_trigger_channels)
    if add_noise:
        channelGenericNoiseAdder.run(
            evt, station, det, amplitude=noise_vrms, min_freq=0 * units.MHz,
            max_freq=max_freq, type='rayleigh')

    rnogHardwareResponse.run(evt, station, det, sim_to_data=True)


def rnog_flower_board_high_low_trigger_simulations(evt, station, det, trigger_channels, trigger_channel_noise_vrms, high_low_trigger_thresholds):
    """ Run the RNO-G FLOWER board high-low trigger simulations.

    This function runs the RNO-G FLOWER board high-low trigger simulations. It performs the following steps:
    - rnogADCResponse: Digitize the traces and run the FLOWER board response
    - highLowThreshold: Apply the high-low trigger thresholds

    Parameters
    ----------
    evt : NuRadioMC.framework.event.Event
        The event to simulate the detector response for.
    station : NuRadioMC.framework.station.Station
        The station to simulate the detector response for.
    det : NuRadioReco.detector.RNO_G.rnog_detector.Detector
        The detector description.
    trigger_channels : list
        The trigger channels (FLOWER) to simulate.
    trigger_channel_noise_vrms : list
        The noise vrms for the trigger channels.
    high_low_trigger_thresholds : dict
        The high-low trigger thresholds for the different trigger rates.

    Returns
    -------
    list
        The noise vrms after the gain for the trigger channels.
    """
    # Runs the FLOWER board response
    vrms_after_gain = rnogADCResponse.run(
        evt, station, det, trigger_channels=trigger_channels,
        vrms=trigger_channel_noise_vrms, digitize_trace=True,
    )

    for idx, trigger_channel in enumerate(trigger_channels):
        logger.debug(
            'Vrms = {:.2f} mV / {:.2f} mV (after gain).'.format(
                trigger_channel_noise_vrms[idx] / units.mV, vrms_after_gain[idx] / units.mV
            ))

    # this is only returning the correct value if digitize_trace=True for self.rnogADCResponse.run(..)
    flower_sampling_rate = station.get_trigger_channel(trigger_channels[0]).get_sampling_rate()
    logger.debug('Flower sampling rate is {:.1f} MHz'.format(
        flower_sampling_rate / units.MHz
    ))

    for thresh_key, threshold in high_low_trigger_thresholds.items():

        if rnogADCResponse.adc_output == "voltage":
            threshold_high = {channel_id: threshold * vrms for channel_id, vrms
                in zip(trigger_channels, vrms_after_gain)}
            threshold_low = {channel_id: -1 * threshold * vrms for channel_id, vrms
                in zip(trigger_channels, vrms_after_gain)}
        else:
            # We round here. This is not how an ADC works but I think this is not needed here.
            threshold_high = {channel_id: int(round(threshold * vrms)) for channel_id, vrms
                in zip(trigger_channels, vrms_after_gain)}
            threshold_low = {channel_id: int(round(-1 * threshold * vrms)) for channel_id, vrms
                in zip(trigger_channels, vrms_after_gain)}

        highLowThreshold.run(
            evt, station, det,
            threshold_high=threshold_high,
            threshold_low=threshold_low,
            use_digitization=False, #the trace has already been digitized with the rnogADCResponse
            high_low_window=6 / flower_sampling_rate,
            coinc_window=20 / flower_sampling_rate,
            number_concidences=2,
            triggered_channels=trigger_channels,
            trigger_name=f"deep_high_low_{thresh_key}",
            pre_trigger_time=250 * units.ns,
        )

    return vrms_after_gain


@functools.lru_cache(maxsize=128)  # this is dangerous if the detector changes it will not notice it!
def get_response_conversion(det, station_id, channel_id, gain_in_dB=3.5):
    """ Get the response conversion between DAQ (RADIANT) and trigger (FLOWER) channel for the given station and channel. """
    radiant_channel = det.get_signal_chain_response(station_id, channel_id, trigger=False)
    flower_channel = det.get_signal_chain_response(station_id, channel_id, trigger=True)

    # radiant = radiant_channel.get("radiant_response")
    radiant_coax = radiant_channel.get("coax_cable")

    flower = flower_channel.get("radiant_response")  # yep we use the same collection name...
    flower_coax = flower_channel.get("coax_cable")

    # we are not using the radiant because we would devide by ~0 out of band
    conversion = flower * flower_coax / radiant_coax

    if gain_in_dB is not None:
        freqs = np.arange(10, 1200, 1) * units.MHz
        gain = np.full_like(freqs, gain_in_dB)
        phase = np.zeros_like(freqs)
        fake_radiant = Response(freqs, [gain, phase], ["dB", "rad"], name="fake_radiant", station_id=-1, channel_id=-1)
        conversion *= fake_radiant

    return conversion


def get_vrms_from_temperature_for_trigger_channels(det, station_id, trigger_channels, temperature):
    """ Get the vrms from the temperature for the trigger channels. """
    vrms_per_channel = []
    for channel_id in trigger_channels:
        resp = det.get_signal_chain_response(station_id, channel_id, trigger=True)
        vrms_per_channel.append(
            signal_processing.calculate_vrms_from_temperature(temperature=temperature, response=resp)
        )

    return np.array(vrms_per_channel)


def get_fiducial_volume(energy):
    # Fiducial volume for a Greenland station.
    # From Martin: https://radio.uchicago.edu/wiki/images/2/26/TriggerSimulation_May2023.pdf

    # key: log10(E), value: radius in km
    max_radius_shallow = {
        16.25: 1.5, 16.5: 2.1, 16.75: 2.7, 17.0: 3.1, 17.25: 3.7, 17.5: 3.9, 17.75: 4.4,
        18.00: 4.8, 18.25: 5.1, 18.50: 5.25, 18.75: 5.3, 19.0: 5.6, 100: 6.1,
    }

    # key: log10(E), value: depth in km
    min_z_shallow = {
        16.25: -0.65, 16.50: -0.8, 16.75: -1.2, 17.00: -1.5, 17.25: -1.7, 17.50: -2.0,
        17.75: -2.1, 18.00: -2.3, 18.25: -2.4, 18.50: -2.55, 100: -2.7,
    }

    def get_limits(dic, E):
        # find all energy bins which are higher than E
        idx = np.arange(len(dic))[E - 10 ** np.array(list(dic.keys())) * units.eV <= 0]
        assert len(idx), f"Energy {E} is too high. Max energy is {10 ** np.amax(dic.keys()):.1e}."

        # take the lowest energy bin which is higher than E
        return np.array(list(dic.values()))[np.amin(idx)] * units.km

    r_max = get_limits(max_radius_shallow, energy)
    z_min = get_limits(min_z_shallow, energy)
    logger.info(f"Cylindric fiducial volume for (lgE = {np.log10(energy):.1f}): "
                f"r_max = {r_max:.2f}m, z_min: {z_min:.2f}m")

    volume = {
        "fiducial_rmax": r_max,
        "fiducial_rmin": 0 * units.km,
        "fiducial_zmin": z_min,
        "fiducial_zmax": 0
    }

    return volume


def RNO_G_HighLow_Thresh(lgRate_per_hz):
    # Thresholds calculated using the RNO-G hardware (iglu + flower_lp)
    # This applies for the VPol antennas
    # parameterization comes from Alan: https://radio.uchicago.edu/wiki/images/e/e6/2023.10.11_Simulating_RNO-G_Trigger.pdf
    return (-859 + np.sqrt(39392706 - 3602500 * lgRate_per_hz)) / 1441.0


if __name__ == "__main__":

    class mySimulation(simulation.simulation):

        def __init__(self, *args, trigger_channel_noise_vrms=None, **kwargs):

            # Read config to get noise type
            tmp_config = simulation.get_config(kwargs["config_file"])

            def wrapper_detector_simulation(*args, **kwargs):
                noise_vrms = signal_processing.calculate_vrms_from_temperature(
                    temperature=tmp_config['trigger']['noise_temperature'],
                    bandwidth=tmp_config["sampling_rate"] / 2)

                detector_simulation(
                    *args, **kwargs, noise_vrms=noise_vrms,
                    max_freq=tmp_config["sampling_rate"] / 2)

            self._detector_simulation_part2 = wrapper_detector_simulation

            super().__init__(*args, **kwargs)

            self.high_low_trigger_thresholds = {
                "10mHz": RNO_G_HighLow_Thresh(-2),
                "100mHz": RNO_G_HighLow_Thresh(-1),
                "1Hz": RNO_G_HighLow_Thresh(0),
                "3Hz": RNO_G_HighLow_Thresh(np.log10(3)),
            }

            assert trigger_channel_noise_vrms is not None, "Please provide the trigger channel noise vrms"
            self.trigger_channel_noise_vrms = trigger_channel_noise_vrms

        def _detector_simulation_filter_amp(self, evt, station, det):
            # apply the amplifiers and filters to get to RADIANT-level
            rnogHardwareResponse.run(evt, station, det, sim_to_data=True)

        def _detector_simulation_trigger(self, evt, station, det):
            vrms_after_gain = rnog_flower_board_high_low_trigger_simulations(
                evt, station, det, deep_trigger_channels, self.trigger_channel_noise_vrms, self.high_low_trigger_thresholds
            )
            for idx, trigger_channel in enumerate(deep_trigger_channels):
                self._Vrms_per_trigger_channel[station.get_id()][trigger_channel] = vrms_after_gain[idx]


    ABS_PATH_HERE = str(os.path.dirname(os.path.realpath(__file__)))
    def_data_dir = os.path.join(ABS_PATH_HERE, "data")
    default_config_path = os.path.join(ABS_PATH_HERE, "../07_RNO_G_simulation/RNO_config.yaml")

    parser = argparse.ArgumentParser(description="Run a NuRadioMC neutrino simulation")

    # General steering arguments
    parser.add_argument("--config", type=str, default=default_config_path, help="Path to a NuRadioMC yaml config file")
    parser.add_argument("--detectordescription", '--det', type=str, default=None,
                        help="Path to a RNO-G detector description file. If None, query the description from hardware database")
    parser.add_argument("--station_id", type=int, default=None, help="Set station to be used in the simulation", required=True)
    parser.add_argument("--proposal", action="store_true",
                        help="Use PROPOSAL to simulate secondaries (only relevant for muon and tau neutrinos with cc interactions)")

    # Neutrino generation arguments. You can either use a file or generate events on the fly.
    # If you use a file, i.e., set --neutrino_file, the following arguments are ignored: --energy, --flavor, --interaction_type
    parser.add_argument("--neutrino_file", type=str, default=None, help="NuRadioMC HDF5 file with neutrino events to be simulated")
    parser.add_argument("--event_list", type=int, default=None, nargs="+",
                        help="Specify event list to be simulated. If not given, all events in the file will be simulated.")
    parser.add_argument("--energy", '-e', default=1e18, type=float, help="Set fixed neutrino energy [eV] (not used if --neutrino_file is set)")
    parser.add_argument("--flavor", '-f', default="all", type=str, choices=["e", "mu", "tau", "all"],
                        help="Choose neutrino flavor to be simulated: e, mu, tau or all (not used if --neutrino_file is set)")
    parser.add_argument("--interaction_type", '-it', default="ccnc", type=str, choices=["cc", "nc", "ccnc"],
                        help="Choose interaction type: cc, nc or ccnc (not used if --neutrino_file is set)")
    parser.add_argument("--n_events", '-n', type=int, default=1e3, help="Number of nu-interactions to be simulated (not used if --neutrino_file is set)")

    # Additonal arguments
    parser.add_argument("--index", '-i', default=0, type=int, help="Counter to create a unique data-set identifier")
    parser.add_argument("--data_dir", type=str, default=def_data_dir, help="Cirectory name where the library will be created")
    parser.add_argument("--nur_output", action="store_true", help="Write nur files.")

    args = parser.parse_args()
    kwargs = args.__dict__
    assert args.station_id is not None, "Please specify a station id with `--station_id`"

    root_seed = secrets.randbits(128)

    det = rnog_detector.Detector(
        detector_file=args.detectordescription, log_level=logging.INFO,
        always_query_entire_description=False, select_stations=args.station_id)

    event_time = dt.datetime(2024, 2, 3)
    det.update(event_time)
    config = simulation.get_config(args.config)

    # Get the trigger channel noise vrms
    trigger_channel_noise_vrms = get_vrms_from_temperature_for_trigger_channels(
        det, args.station_id, deep_trigger_channels, config['trigger']['noise_temperature'])

    logger.info(f"Trigger channel noise vrms (used for the definition of the trigger threshold): {np.around(trigger_channel_noise_vrms / units.mV, 2)} mV")

    # Simulate fiducial volume around station
    volume = get_fiducial_volume(args.energy)
    pos = det.get_absolute_position(args.station_id)
    logger.info(f"Simulating around center x0={pos[0]:.2f}m, y0={pos[1]:.2f}m")
    volume.update({"x0": pos[0], "y0": pos[1]})

    output_path = f"{args.data_dir}/station_{args.station_id}/nu_{args.flavor}_{args.interaction_type}"

    if args.neutrino_file is not None:
        output_path += f"/{os.path.basename(args.neutrino_file).replace('.hdf5', '')}"

    if not os.path.exists(output_path):
        logger.debug(f"Create output directory: {output_path}")
        os.makedirs(output_path, exist_ok=True)

    output_filename = (f"{output_path}/{args.flavor}_{args.interaction_type}"
                       f"_1e{np.log10(args.energy):.2f}eV_{args.index:08d}.hdf5")

    flavor_ids = {"e": [12, -12], "mu": [14, -14], "tau": [16, -16], "all": [12, 14, 16, -12, -14, -16]}
    run_proposal = args.proposal and ("cc" in args.interaction_type) and (args.flavor in ["mu", "tau", "all"])
    if run_proposal:
        logger.info(f"Using PROPOSAL for simulation of {args.flavor} {args.interaction_type}")

    if args.neutrino_file is None:
        input_data = generator.generate_eventlist_cylinder(
            "on-the-fly",
            kwargs["n_events"],
            args.energy, args.energy,
            volume,
            start_event_id=args.index * args.n_events + 1,
            flavor=flavor_ids[args.flavor],
            n_events_per_file=None,
            deposited=False,
            proposal=run_proposal,
            proposal_config="Greenland",
            start_file_id=0,
            log_level=None,
            proposal_kwargs={},
            max_n_events_batch=args.n_events,
            write_events=False,
            seed=root_seed + args.index,
            interaction_type=args.interaction_type,
        )
    else:
        input_data = args.neutrino_file
        if not os.path.exists(input_data):
            raise FileNotFoundError(f"Input file {input_data} does not exist")
        logger.info(f"Read neutrino interactions from input file: {input_data}")
        if args.event_list is not None:
            logger.info(f"Only simulate events {args.event_list} from input file")

    if args.nur_output:
        nur_output_filename = output_filename.replace(".hdf5", ".nur")
    else:
        nur_output_filename = None

    sim = mySimulation(
        inputfilename=input_data,
        outputfilename=output_filename,
        det=det,
        evt_time=event_time,
        outputfilenameNuRadioReco=nur_output_filename,
        config_file=args.config,
        trigger_channels=deep_trigger_channels,
        trigger_channel_noise_vrms=trigger_channel_noise_vrms,
        event_list=args.event_list,
        use_cpp=True,
    )

    sim.run()
