#!/bin/env python3

from NuRadioReco.framework.event import Event
from NuRadioReco.framework.station import Station
from NuRadioReco.framework.channel import Channel

from NuRadioReco.utilities import units, fft, signal_processing
from NuRadioReco.detector.RNO_G import rnog_detector
from NuRadioReco.modules.RNO_G import hardwareResponseIncorporator, triggerBoardResponse
from NuRadioReco.modules.trigger import highLowThreshold
import NuRadioReco.modules.channelGenericNoiseAdder

from matplotlib import pyplot as plt
from collections import defaultdict
from scipy import constants
import datetime as dt
import numpy as np
import argparse
import copy
import time
import json
import sys


import NuRadioMC  # to init the logger
import logging
logger = logging.getLogger("NuRadioMC.noise_rate_simulation")
logger.setLevel(logging.INFO)

"""
The purpose of this script is to simulate the noise trigger rate for the RNO-G deep trigger.
This script is different from the one in NuRadioReco/utilities/noise.py in that it allways
simulates the 4 trigger channel traces and runs a trigger simulation on them. This allows to
estimate the trigger rate but is not the most efficient way to simulate triggerd noise!

This script is optiumized for speed which is why it is not using always the plain NuRadioReco
modules to do certain operations. The idea is to minimize the amount of Fourier transforms
as much as possible. This means we create noise in the Fourier domain, apply the filters in the
Fourier domain and only transform back to the time domain when running "trigger_simulation".

While a few settings are hard-coded at the beginning of the script, some other parameters are
passed as arguments.

This script produces a json file with the total simulation time, the number of triggers per trigger type
and a few other meta data which are necessary to interpret the results.
"""

# define the deep trigger channels
deep_trigger_channels = np.array([0, 1, 2, 3])

# Define the ADC output. Options are "voltage" or "counts". The latter is more realistic.
# If you set it to "voltage" the traces to trigger on are discrete voltages. The thresholds
# are __not__ discretized. If you set it to "counts" the traces are digitized and the thresholds
# are in counts (discrete integer values).
adc_output = "counts"

# Initialize modules
highLowThresholdTrigger = highLowThreshold.triggerSimulator()
triggerBoard = triggerBoardResponse.triggerBoardResponse(log_level=logging.INFO)
triggerBoard.begin(clock_offset=0, adc_output=adc_output)

hardwareResponse = hardwareResponseIncorporator.hardwareResponseIncorporator()
hardwareResponse.begin(trigger_channels=deep_trigger_channels)

# Define the thresholds for the trigger simulation
sigma_thresholds = np.linspace(3.2, 4, 20)
high_low_trigger_thresholds = {s: s for s in sigma_thresholds}


def get_vrms_per_channel(args, noise_kwargs, det, filters):
    """
    Get the vrms of the noise traces for each channel.
    This function is only used to calculate and plot the vrms noise distribution per channel.
    """
    # It is important to create a new noiseAdder for each run (when using ray). Otherwise the noise is not random.
    noiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()

    vrms_per_channel = defaultdict(list)
    for _ in range(args.nevents):
        event, station = get_noise_event(
            noiseAdder, det, deep_trigger_channels, noise_kwargs)

        detector_simulation(event, station, filters, det)

        for channel in station.iter_trigger_channels():
            vrms_per_channel[channel.get_id()].append(np.std(channel.get_trace()))

    fig, ax = plt.subplots()

    all_vrms = np.hstack(list(vrms_per_channel.values()))
    vrms_per_channel = {channel_id: np.array(vrms) for channel_id, vrms in vrms_per_channel.items()}
    h, bins = np.histogram(all_vrms / units.mV, bins=30)
    for channel_id, vrms in vrms_per_channel.items():
        print(f"Channel {channel_id}: {np.mean(vrms) / units.mV:.2f} mV")
        ax.hist(vrms / units.mV, bins=bins, histtype='step', label=f'Channel {channel_id}')

    ax.set_xlabel('vrms / mV')
    ax.legend()
    fig.tight_layout()
    plt.show()

    return vrms_per_channel


def get_noise_event(noiseAdder, det, channel_ids, noise_kwargs):
    """ Create a noise event with the given noise parameters.

    Parameters
    ----------
    noiseAdder: NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder
        The noise adder module
    det: NuRadioReco.detector.RNO_G.rnog_detector.Detector
        The detector object
    channel_ids: list of int
        The channel ids to simulate noise for
    noise_kwargs: dict
        The noise parameters

    Returns
    -------
    event: NuRadioReco.framework.event.Event
        The event object
    station: NuRadioReco.framework.station.Station
        The station object
    """

    station_ids = det.get_station_ids()
    assert len(station_ids) == 1, "Only one station is supported for this simulation"
    station_id = station_ids[0]

    event = Event(0, 0)
    station = Station(station_id)

    for channel_id in channel_ids:
        channel = Channel(channel_id)
        det_channel = det.get_channel(station_id, channel_id)
        sampling_rate = det_channel["trigger_adc_sampling_frequency"]

        # we apply the hardware response (filter) to the noise traces afterwards
        # Thus the amplitude can be given for a flat rectangular filter/bandwidth
        # as specified with `min_freq` and `max_freq`

        noise_spec = noiseAdder.bandlimited_noise(
            sampling_rate=sampling_rate, type='rayleigh', time_domain=False, **noise_kwargs)

        channel.set_frequency_spectrum(noise_spec, sampling_rate)
        station.add_channel(channel)

    event.set_station(station)

    return event, station


def detector_simulation(evt, station, filters, det):
    """ Simulate the detector response.

    For speed reasons we pass a list of the filters as complex numpy arrays
    to apply the channel response instead of calling the hardwareResponseIncorperator.

    Parameters
    ----------
    evt: NuRadioReco.framework.event.Event
        The event object
    station: NuRadioReco.framework.station.Station
        The station object
    filters: dict
        The filters for the deep trigger channels
    det: NuRadioReco.detector.RNO_G.rnog_detector.Detector
        The detector object
    """
    # apply the amplifiers and filters to get to RADIANT-level
    for channel in station.iter_channels():
        channel.set_frequency_spectrum(
            channel.get_frequency_spectrum() * filters[channel.get_id()], "same")

    # hardwareResponse.run(evt, station, det, sim_to_data=True)

def trigger_simulation(evt, station, det, vrms=None):
    """ Simulate the trigger response.

    This function does two things:
        1. It runs the triggerBoardResponse module to simulate the FLOWER board response.
        This includes the gain equaliztion and analog to digital conversion.
        2. It runs the highLowThreshold module to simulate the high-low trigger. YOU CAN
        MODIFY THIS PART IF YOU WANT TO SIMULATE A DIFFERENT TRIGGER.

    Parameters
    ----------
    evt: NuRadioReco.framework.event.Event
        The event object
    station: NuRadioReco.framework.station.Station
        The station object
    det: NuRadioReco.detector.RNO_G.rnog_detector.Detector
        The detector object
    vrms: np.ndarray
        The vrms of the noise traces per channel

    Returns
    -------
    vrms_after_gain: np.ndarray
        The vrms of the noise traces after gain equalization
    """

    # Runs the FLOWER board response
    vrms_after_gain = triggerBoard.run(
        evt, station, det, trigger_channels=deep_trigger_channels,
        vrms=vrms, apply_adc_gain=True, digitize_trace=True,
    )
    logger.debug(f'VRMS after gain: {vrms_after_gain} ({vrms})')

    # this is only returning the correct value if digitize_trace=True for rnogADCResponse.run(..)
    flower_sampling_rate = station.get_trigger_channel(deep_trigger_channels[0]).get_sampling_rate()
    logger.debug(f'Flower sampling rate is {flower_sampling_rate / units.MHz:.1f} MHz')


    for thresh_key, threshold in high_low_trigger_thresholds.items():
        if adc_output == "voltage":
            threshold_high = {channel_id: threshold * vrms for channel_id, vrms
                in zip(deep_trigger_channels, vrms_after_gain)}
            threshold_low = {channel_id: -1 * threshold * vrms for channel_id, vrms
                in zip(deep_trigger_channels, vrms_after_gain)}
        else:
            # We round here. This is not how an ADC works but I think this is not needed here.
            threshold_high = {channel_id: int(round(threshold * vrms)) for channel_id, vrms
                in zip(deep_trigger_channels, vrms_after_gain)}
            threshold_low = {channel_id: int(round(-1 * threshold * vrms)) for channel_id, vrms
                in zip(deep_trigger_channels, vrms_after_gain)}

        highLowThresholdTrigger.run(
            evt,
            station,
            det,
            threshold_high=threshold_high,
            threshold_low=threshold_low,
            use_digitization=False, #the trace has already been digitized with the rnogADCResponse
            high_low_window=6 / flower_sampling_rate,
            coinc_window=20 / flower_sampling_rate,
            number_concidences=2,
            triggered_channels=deep_trigger_channels,
            trigger_name=f"deep_high_low_{thresh_key:.4f}_sigma",
        )

    return vrms_after_gain


def process(det, filters, n_events, noise_kwargs, noiseAdder=None):
    """ Run the simulation for a given number of events. """
    noise_kwargs = copy.deepcopy(noise_kwargs)
    triggers = defaultdict(list)
    t0 = time.time()

    # It is important to create a new noiseAdder for each run (when using ray). Otherwise the noise is not random.
    noiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
    seed = np.random.randint(0, 2**32)
    noiseAdder.begin(seed=seed)

    # take out the argument again which is not meant to be used for the
    # noise adder module.
    vrms_per_channel = noise_kwargs.pop("vrms_per_channel", None)

    for _ in range(n_events):
        event, station = get_noise_event(
            noiseAdder, det, deep_trigger_channels, noise_kwargs)

        detector_simulation(event, station, filters, det)
        vrms = trigger_simulation(event, station, det, vrms=vrms_per_channel)

        for trigger_name, trigger in station.get_triggers().items():
            triggers[trigger_name].append(int(trigger.has_triggered()))

        triggers["vrms"].append(vrms.tolist())

    t_tot = time.time() - t0
    logger.info(f"Simulation of {n_events} events took {t_tot:.2f}s ({t_tot / n_events:.4f}s per event)")

    return triggers


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
    parser.add_argument("--n_samples", type=int, default=32768, help="")
    parser.add_argument("--index", type=int, default=0, help="")
    parser.add_argument("--ray", action="store_true")
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--running_vrms", action="store_true")

    args = parser.parse_args()

    if args.label != "":
        args.label = f"_{args.label}"

    logger.info(f"Simulate the noise trigger rate for station {args.station_id} "
                f"(using channels {deep_trigger_channels})")

    det = rnog_detector.Detector(
        detector_file=args.detectordescription, log_level=logging.INFO,
        always_query_entire_description=True, select_stations=args.station_id)

    det.update(dt.datetime(2023, 8, 3))

    max_freq = det.get_sampling_frequency(args.station_id, None, trigger=True) / 2
    bandwidth = np.array([50, max_freq / units.MHz]) * units.MHz

    # Calculate the vrms for the given temperature and bandwidth. It is _CORRECT_
    # to not account for the response of the channel here.
    vrms = signal_processing.calculate_vrms_from_temperature(300 * units.kelvin, bandwidth)
    logger.info(f"VRMS [for bandwidth {bandwidth / units.MHz} MHz]: {vrms / units.microvolt:.2f} muV")
    noise_kwargs = {
        "amplitude": vrms,
        "min_freq": bandwidth[0],
        "max_freq": bandwidth[1],
        "n_samples": args.n_samples,
    }

    freqs = fft.freqs(noise_kwargs['n_samples'], det.get_sampling_frequency(args.station_id, None, trigger=True))
    filters = {channel_id: det.get_signal_chain_response(args.station_id, channel_id, trigger=True)(freqs)
        for channel_id in deep_trigger_channels}

    if not args.running_vrms:
        # this might return slightly different results than the function get_vrms_per_channel
        # because of the frequency resolution with which the spectra are sampled.
        vrms_per_channel = np.array(
            [signal_processing.calculate_vrms_from_temperature(
                300 * units.kelvin,
                response=det.get_signal_chain_response(args.station_id, channel_id, trigger=True))
            for channel_id in deep_trigger_channels]
        )

        # We are highjacking this dict to carry arguments for the triggerBoardResponse module
        # This vrms (with amplification) is used to calculate the FLOWER gain. This argument
        # also effects the realized trigger thresholds.
        logger.info(f"VRMS per channel (incl. amplifier): {np.around(vrms_per_channel / units.mV, 2)} mV")
        noise_kwargs["vrms_per_channel"] = vrms_per_channel

    n_events = args.nevents
    t0 = time.time()
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

        if args.nruns // args.ncpus > 2:
            print("Warning: You are using more than 2 runs per CPU. This might be inefficient.")
        ray.init(num_cpus=args.ncpus)
        @ray.remote
        def process_ray(*args, **kwargs):
            return process(*args, **kwargs)

        det_ref = ray.put(det)
        filters_ref = ray.put(filters)
        n_events_ref = ray.put(n_events)
        noise_kwargs_ref = ray.put(noise_kwargs)

        remotes = [process_ray.remote(det_ref, filters_ref, n_events_ref, noise_kwargs_ref)
            for _ in range(args.nruns)]
        outs = ray.get(remotes)
    else:
        outs = [process(det, filters, n_events, noise_kwargs) for _ in range(args.nruns)]

    triggers = defaultdict(list)
    for out in outs:
        for trigger_name, trigger_res in out.items():
            triggers[trigger_name].extend(trigger_res)

    assert len(triggers[trigger_name]) == n_events * args.nruns, "Number of events does not match"

    vrms = triggers.pop("vrms", None)
    if vrms is not None:
        vrms = np.array(vrms)
        vrms = [np.mean(vrms, axis=0).tolist(), np.std(vrms, axis=0).tolist()]
        if adc_output == "counts":
            logger.info(
                f"VRMS (gain equalized): {np.around(vrms[0], 2)} +- {np.around(vrms[1], 2)} ADC")
        else:
            logger.info(
                f"VRMS (gain equalized): {np.around(vrms[0] / units.mV, 2)} +- "
                f"{np.around(vrms[1] / units.mV, 2)} mV")


    dt = noise_kwargs["n_samples"] / det.get_sampling_frequency(args.station_id, None, trigger=True)
    total_time = dt * (n_events * args.nruns)
    print(f"Total time simulated: {total_time / units.s:.1f}s ({n_events * args.nruns} events)")
    tsim = time.time() - t0
    print(f"Total simulation time: {tsim:.2f}s ({tsim / (n_events * args.nruns):.4f}s per event)")

    max_rate = 1 / dt

    data = {
        "vrms": vrms,
        "total_time": total_time,
        "tot_nevents": n_events * args.nruns
    }

    for trigger_name, trigger_data in triggers.items():
        threshold = float(trigger_name.split("_")[-2])

        n_triggers = int(np.sum(trigger_data))
        efficiency = n_triggers / len(trigger_data)
        trigger_rate = n_triggers / total_time

        if trigger_rate > max_rate:
            logger.warning(f"Trigger rate for {trigger_name} is higher than max trigger rate ({max_rate / units.Hz:.2f}): "
                            f"{trigger_rate / units.Hz:.2f} Hz")

        e_trigger_rate = np.sqrt(n_triggers) / total_time
        data[trigger_name] = {"n_triggers": n_triggers, "threshold_sigma": threshold}

        logger.info(f"Trigger efficiency for {trigger_name.replace('deep_high_low_', '')}: "
                    f"{efficiency:.3e} ({n_triggers}) "
                    f"({trigger_rate / units.Hz:.2f} +- {e_trigger_rate / units.Hz:.2f} Hz)")

    vrms_label = 'running_vrms' if args.running_vrms else 'fixed_vrms'
    with open(f"trigger_rates_st{args.station_id}_{adc_output}_{vrms_label}_{noise_kwargs['n_samples']}_{sigma_thresholds[0]:.2f}-"
              f"{sigma_thresholds[-1]:.2f}_{total_time / units.s:.1f}s{args.label}.json", "w") as f:
        json.dump(data, f)
