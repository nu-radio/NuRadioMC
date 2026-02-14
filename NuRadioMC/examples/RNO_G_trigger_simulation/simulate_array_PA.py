#!/bin/env python3

import argparse
import numpy as np
import os
import secrets
import functools
import datetime as dt
import time
from numpy.random import Generator, Philox

from NuRadioMC.EvtGen import generator
from NuRadioMC.simulation import simulation
from NuRadioReco.utilities import units, signal_processing, fft

from NuRadioReco.detector.RNO_G import rnog_detector
from NuRadioReco.detector.response import Response

import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.channelGenericNoiseAdder

from NuRadioReco.modules.RNO_G import hardwareResponseIncorporator, triggerBoardResponse
from NuRadioReco.modules.trigger import highLowThreshold
from NuRadioReco.modules.phasedarray import beamformedPowerIntegrationTrigger

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

powerTrigger = beamformedPowerIntegrationTrigger.BeamformedPowerIntegrationTrigger()


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
    
    power_thresh = 0
    for ch in range(4):
        power_thresh+=vrms_after_gain[ch]**2

    powerTrigger.run(
        evt, station, det, Vrms=None, trigger_name="phased_array", threshold=np.rint(power_thresh*9), triggered_channels=[0,1,2,3],
        phasing_angles=np.arcsin(np.linspace(np.sin(np.deg2rad(60)),np.sin(np.deg2rad(-60)),12)),
        ref_index=1.75, trigger_adc=True, apply_digitization=False, adc_output="counts", upsampling_factor=4,
        upsampling_method="fir", window=24, averaging_divisor=32, step=4, saturation_bits=8, filter_taps=45,
        coeff_gain=256 )


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
    
    # Get RADIANT sampling rate from regular channel (for jitter calculation)
    radiant_sampling_rate = station.get_channel(trigger_channels[0]).get_sampling_rate()
    logger.debug('RADIANT sampling rate is {:.1f} GHz'.format(
        radiant_sampling_rate / units.GHz
    ))
    # Create jitter for pre_trigger_time and apply to phased_array trigger
    # Use a single float value so it applies to ALL channels (not just trigger channels)
    root_seed = secrets.randbits(128)
    rng = Generator(Philox(root_seed))
    
    def _jitter_adder_internal(sample_rate, rnd):
        """
        Add random jitter to trigger time.
        In RNO-G data, every data is collected in 64 sample blocks then sent to the trigger,
        so the trigger time will vary randomly within ±64 samples. Uses RADIANT sampling rate (2.4 GHz).

        Note: this module has not been tested based on correlation between the jitter channels yet.

        Parameters
        ----------
        sample_rate : float
            Sampling rate in Hz (should be RADIANT rate ~2.4 GHz)
        rnd : numpy.random.Generator
            Random number generator. 
        
        Returns
        -------
        float
            Jitter time in seconds
        """
        if rnd is None:
            breakpoint()


        low = -64
        high = 64
        jitter_time = rnd.uniform(low, high) / sample_rate * units.Hz
        return jitter_time

    
    # Create pre_trigger_times for ALL channels (ch0-ch23) to avoid KeyError during readout
    # Use RADIANT sampling rate (2.4 GHz) for jitter calculation
    phased_array_pre_trigger_dict = {
        channel_id: 250 * units.ns + _jitter_adder_internal(radiant_sampling_rate, rng) 
        for channel_id in range(24)
    }

    # Set pre_trigger_times on the phased_array trigger (used for readout window calculation)
    station.get_trigger("phased_array").set_pre_trigger_times(phased_array_pre_trigger_dict)

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

        # Use RADIANT sampling rate (2.4 GHz) for jitter calculation
        pre_trigger_dict = {channel_id: 250 * units.ns + _jitter_adder_internal(radiant_sampling_rate, rng) for channel_id in trigger_channels}

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
            pre_trigger_time=pre_trigger_dict,
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


def get_array_fiducial_volume(energy, det, station_ids):
    """
    Get the fiducial volume that encompasses all stations in the array.
    
    Parameters
    ----------
    energy : float
        Neutrino energy in eV
    det : Detector
        Detector description object
    station_ids : list of int
        List of station IDs to include
        
    Returns
    -------
    dict
        Dictionary with volume parameters (x0, y0, fiducial_rmin, fiducial_rmax, fiducial_zmin, fiducial_zmax)
    """
    # Get station positions
    station_positions = []
    for station_id in station_ids:
        pos = det.get_absolute_position(station_id)
        station_positions.append(pos[:2])  # x, y only
    
    station_positions = np.array(station_positions)
    
    # Calculate center of array (mean position)
    center_x = np.mean(station_positions[:, 0])
    center_y = np.mean(station_positions[:, 1])
    
    # Calculate maximum distance from center to any station
    distances_from_center = np.sqrt((station_positions[:, 0] - center_x)**2 + 
                                     (station_positions[:, 1] - center_y)**2)
    max_station_distance = np.max(distances_from_center)
    
    # Get base fiducial volume for single station
    single_volume = get_fiducial_volume(energy)
    single_radius = single_volume["fiducial_rmax"]
    
    # Total radius should be: distance to furthest station + single station radius
    total_radius = max_station_distance + single_radius
    
    logger.info(f"Array fiducial volume centered at x0={center_x:.2f}m, y0={center_y:.2f}m")
    logger.info(f"  Max station distance from center: {max_station_distance:.2f}m")
    logger.info(f"  Single station radius: {single_radius:.2f}m")
    logger.info(f"  Total array radius: {total_radius:.2f}m")
    
    volume = {
        "x0": center_x,
        "y0": center_y,
        "fiducial_rmax": total_radius,
        "fiducial_rmin": 0 * units.km,
        "fiducial_zmin": single_volume["fiducial_zmin"],
        "fiducial_zmax": single_volume["fiducial_zmax"]
    }
    
    return volume


def RNO_G_HighLow_Thresh(lgRate_per_hz):
    # Thresholds calculated using the RNO-G hardware (iglu + flower_lp)
    # This applies for the VPol antennas
    # parameterization comes from Alan: https://radio.uchicago.edu/wiki/images/e/e6/2023.10.11_Simulating_RNO-G_Trigger.pdf
    return (-859 + np.sqrt(39392706 - 3602500 * lgRate_per_hz)) / 1441.0


if __name__ == "__main__":
    # Start timing the simulation
    start_time = time.time()
    logger.info("Starting RNO-G array simulation...")

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
                "3.8sigma": 3.8,
                "4.0sigma": 4.0,
                "4.2sigma": 4.2
            }

            assert trigger_channel_noise_vrms is not None, "Please provide the trigger channel noise vrms"
            self.trigger_channel_noise_vrms = trigger_channel_noise_vrms

        def _detector_simulation_filter_amp(self, evt, station, det):
            # apply the amplifiers and filters to get to RADIANT-level
            rnogHardwareResponse.run(evt, station, det, sim_to_data=True)

        def _detector_simulation_trigger(self, evt, station, det):
            # Get the noise vrms for this specific station
            station_id = station.get_id()
            vrms_after_gain = rnog_flower_board_high_low_trigger_simulations(
                evt, station, det, deep_trigger_channels, 
                self.trigger_channel_noise_vrms[station_id], 
                self.high_low_trigger_thresholds
            )
            for idx, trigger_channel in enumerate(deep_trigger_channels):
                self._Vrms_per_trigger_channel[station_id][trigger_channel] = vrms_after_gain[idx]


    ABS_PATH_HERE = str(os.path.dirname(os.path.realpath(__file__)))
    def_data_dir = os.path.join(ABS_PATH_HERE, "data_array")
    default_config_path = os.path.join(ABS_PATH_HERE, "../config.yaml")  # Config in parent directory

    parser = argparse.ArgumentParser(description="Run a NuRadioMC neutrino simulation for the whole RNO-G array")

    # General steering arguments
    parser.add_argument("--config", type=str, default=default_config_path, help="Path to a NuRadioMC yaml config file")
    parser.add_argument("--detectordescription", '--det', type=str, default=None,
                        help="Path to a RNO-G detector description file. If None, query the description from hardware database")
    parser.add_argument("--station_ids", type=int, nargs="+", default=[11, 12, 13, 14, 21, 22, 23, 24],
                        help="List of station IDs to simulate (default: 11 12 13 14 21 22 23 24)")
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
    parser.add_argument("--data_dir", type=str, default=def_data_dir, help="Directory name where the library will be created")
    parser.add_argument("--nur_output", action="store_true", help="Write nur files.")

    args = parser.parse_args()
    kwargs = args.__dict__

    root_seed = secrets.randbits(128)

    # Initialize detector with all requested stations (or None to get all commissioned stations)
    logger.info(f"Initializing detector with stations: {args.station_ids}")
    det = rnog_detector.Detector(
        detector_file=args.detectordescription, log_level=logging.INFO,
        always_query_entire_description=False, select_stations=args.station_ids)

    event_time = dt.datetime(2024, 2, 3)
    #event_time = dt.datetime(2024, 8, 14)
    det.update(event_time)
    config = simulation.get_config(args.config)
    logger.info("Config file: %s" % args.config)
    logger.info("Cross section type: %s" % config['weights']['cross_section_type'])

    # Get the trigger channel noise vrms for all stations
    trigger_channel_noise_vrms = {}
    for station_id in args.station_ids:
        trigger_channel_noise_vrms[station_id] = get_vrms_from_temperature_for_trigger_channels(
            det, station_id, deep_trigger_channels, config['trigger']['noise_temperature'])
        logger.info(f"Station {station_id} - Trigger channel noise vrms: {np.around(trigger_channel_noise_vrms[station_id] / units.mV, 2)} mV")

    # Calculate array fiducial volume
    volume = get_array_fiducial_volume(args.energy, det, args.station_ids)

    # Set output path for array simulation
    output_path = f"{args.data_dir}/array_stations_{'_'.join(map(str, args.station_ids))}/nu_{args.flavor}_{args.interaction_type}"


    if args.neutrino_file is not None:
        output_path += f"/{os.path.basename(args.neutrino_file).replace('.hdf5', '')}"

    if not os.path.exists(output_path):
        logger.debug(f"Create output directory: {output_path}")
        os.makedirs(output_path, exist_ok=True)

    output_filename = (f"{output_path}/{args.flavor}_{args.interaction_type}"
                       f"_1e{np.log10(args.energy):.2f}eV_{args.index:08d}.hdf5")

    flavor_ids = {
        "e": np.array([12, -12]),
        "mu": np.array([14, -14]),
        "tau": np.array([16, -16]),
        "all": np.array([12, 14, 16, -12, -14, -16])
    }
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
            cross_sections_model='ctw'
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
    
    # End timing and print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Format time in a human-readable way
    hours = int(execution_time // 3600)
    minutes = int((execution_time % 3600) // 60)
    seconds = execution_time % 60
    
    if hours > 0:
        time_str = f"{hours}h {minutes}m {seconds:.2f}s"
    elif minutes > 0:
        time_str = f"{minutes}m {seconds:.2f}s"
    else:
        time_str = f"{seconds:.2f}s"
    
    print(f"RNO-G array simulation completed successfully!")
    print(f"Total execution time: {time_str} ({execution_time:.2f} seconds)")
    print(f"Simulated {len(args.station_ids)} stations: {args.station_ids}")

