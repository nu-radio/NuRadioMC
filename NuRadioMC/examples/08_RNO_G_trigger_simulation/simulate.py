#!/bin/env python3

import argparse
import copy
import logging
import numpy as np
import os
import secrets
import functools
import datetime as dt
from scipy import constants



from NuRadioMC.EvtGen import generator
from NuRadioMC.simulation import simulation
from NuRadioReco.utilities import units

from NuRadioReco.detector.RNO_G import rnog_detector

from NuRadioReco.modules.RNO_G import hardwareResponseIncorporator, triggerBoardResponse
from NuRadioReco.modules.trigger import highLowThreshold


root_seed = secrets.randbits(128)
deep_trigger_channels = np.array([0, 1, 2, 3])

@functools.lru_cache(maxsize=128)  # this is dangerous if the detector changes it will not notice it!
def get_response_conversion(det, station_id, channel_id):
    radiant_channel = det.get_signal_chain_response(station_id, channel_id, trigger=False)
    flower_channel = det.get_signal_chain_response(station_id, channel_id, trigger=True)

    radiant = radiant_channel.get("radiant_response")
    radiant_coax = radiant_channel.get("coax_cable")

    flower = flower_channel.get("radiant_response")  # yep we use the same collection name...
    flower_coax = flower_channel.get("coax_cable")

    conversion = flower * flower_coax / (radiant * radiant_coax)

    return conversion


def get_vrms_from_temperature_for_trigger_channels(det, station_id, trigger_channels, temperature):

    vrms_per_channel = []
    for channel_id in trigger_channels:
        resp = det.get_signal_chain_response(station_id, channel_id, trigger=True)

        freqs = np.linspace(10, 1200, 1000) * units.MHz
        filt = resp(freqs)

        # Calculation of Vrms. For details see from elog:1566 and https://en.wikipedia.org/wiki/Johnson%E2%80%93Nyquist_noise
        # (last two Eqs. in "noise voltage and power" section) or our wiki https://nu-radio.github.io/NuRadioMC/NuRadioMC/pages/HDF5_structure.html

        # Bandwidth, i.e., \Delta f in equation
        integrated_channel_response = np.trapz(np.abs(filt) ** 2, freqs)

        vrms_per_channel.append(
            (temperature * 50 * constants.k * integrated_channel_response / units.Hz) ** 0.5
        )

    return vrms_per_channel


def get_fiducial_volume(energy):
    # Fiducial volume for a Greenland station. From Martin: https://radio.uchicago.edu/wiki/images/2/26/TriggerSimulation_May2023.pdf

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
    print(f"Maximum radius {r_max}")
    z_min = get_limits(min_z_shallow, energy)
    print(f"Depth {z_min}")

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


class mySimulation(simulation.simulation):

    def __init__(self, *args, **kwargs):
        # this module is needed in super().__init__ to calculate the vrms
        self.rnogHarwareResponse = hardwareResponseIncorporator.hardwareResponseIncorporator()
        self.rnogHarwareResponse.begin(trigger_channels=deep_trigger_channels)

        # Read config to get noise type
        tmp_config = simulation.get_config(kwargs["config_file"])
        if tmp_config["noise_type"] == "data_driven":
            self._detector_simulation_part2 = self.detector_simulation_with_data_driven_noise


        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("NuRadioMC.RNOG_trigger_simulation")
        self.deep_trigger_channels = deep_trigger_channels


        self.highLowThreshold = highLowThreshold.triggerSimulator()
        self.rnogADCResponse = triggerBoardResponse.triggerBoardResponse()
        self.rnogADCResponse.begin(
            clock_offset=0.0, adc_output="counts")

        # future TODO: Add noise
        # self.channel_generic_noise_adder = channelGenericNoiseAdder.channelGenericNoiseAdder()
        # self.channel_generic_noise_adder.begin(seed=self._cfg['seed'])

        self.output_mode = {'Channels': self._config['output']['channel_traces'],
                            'ElectricFields': self._config['output']['electric_field_traces'],
                            'SimChannels': self._config['output']['sim_channel_traces'],
                            'SimElectricFields': self._config['output']['sim_electric_field_traces']}

        self.high_low_trigger_thresholds = {
            "10mHz": RNO_G_HighLow_Thresh(-2),
            "100mHz": RNO_G_HighLow_Thresh(-1),
            "1Hz": RNO_G_HighLow_Thresh(0),
            "3Hz": RNO_G_HighLow_Thresh(np.log10(3)),
        }


    def _detector_simulation_filter_amp(self, evt, station, det):
        # apply the amplifiers and filters to get to RADIANT-level
        self.rnogHarwareResponse.run(evt, station, det, sim_to_data=True)

    def _detector_simulation_trigger(self, evt, station, det):

        vrms_input_to_adc = get_vrms_from_temperature_for_trigger_channels(
            det, station.get_id(), self.deep_trigger_channels, 300)

        sampling_rate = det.get_sampling_frequency(station.get_id())
        self.logger.info(f'Radiant sampling rate is {sampling_rate / units.MHz:.1f} MHz')

        # Runs the FLOWER board response
        vrms_after_gain = self.rnogADCResponse.run(
            evt, station, det, trigger_channels=self.deep_trigger_channels,
            vrms=copy.copy(vrms_input_to_adc), digitize_trace=True,
        )

        for idx, trigger_channel in enumerate(self.deep_trigger_channels):
            self.logger.info(
                f'Vrms = {vrms_input_to_adc[idx] / units.mV:.2f} mV / {vrms_after_gain[idx] / units.mV:.2f} mV (after gain).')
            self._Vrms_per_trigger_channel[station.get_id()][trigger_channel] = vrms_after_gain[idx]


        # this is only returning the correct value if digitize_trace=True for self.rnogADCResponse.run(..)
        flower_sampling_rate = station.get_trigger_channel(self.deep_trigger_channels[0]).get_sampling_rate()
        self.logger.info(f'Flower sampling rate is {flower_sampling_rate / units.MHz:.1f} MHz')

        for thresh_key, threshold in self.high_low_trigger_thresholds.items():

            if self.rnogADCResponse.adc_output == "voltage":
                threshold_high = {channel_id: threshold * vrms for channel_id, vrms
                    in zip(self.deep_trigger_channels, vrms_after_gain)}
                threshold_low = {channel_id: -1 * threshold * vrms for channel_id, vrms
                    in zip(self.deep_trigger_channels, vrms_after_gain)}
            else:
                # We round here. This is not how an ADC works but I think this is not needed here.
                threshold_high = {channel_id: int(round(threshold * vrms)) for channel_id, vrms
                    in zip(self.deep_trigger_channels, vrms_after_gain)}
                threshold_low = {channel_id: int(round(-1 * threshold * vrms)) for channel_id, vrms
                    in zip(self.deep_trigger_channels, vrms_after_gain)}

            self.highLowThreshold.run(
                evt,
                station,
                det,
                threshold_high=threshold_high,
                threshold_low=threshold_low,
                use_digitization=False, #the trace has already been digitized with the rnogADCResponse
                high_low_window=6 / flower_sampling_rate,
                coinc_window=20 / flower_sampling_rate,
                number_concidences=2,
                triggered_channels=self.deep_trigger_channels,
                trigger_name=f"deep_high_low_{thresh_key}",
            )

    def detector_simulation_with_data_driven_noise(self, evt, station, det):

        # convolve efield with antenna pattern and add cable delay (this is also done in the efieldToVoltageConverter
        # (unlike the efieldToVoltageConverterPEREFIELD))
        self.efieldToVoltageConverter.run(evt, station, det)

        self._detector_simulation_filter_amp(evt, station, det)

        for channel in station.get_channels():
            channel_id = channel.get_id()
            noise_spec = self.channelGenericNoiseAdder.bandlimited_noise(
                min_freq=10 * units.MHz, max_freq=1000 * units.MHz,
                n_samples=channel.get_number_of_samples(), sampling_rate=channel.get_sampling_rate(),
                amplitude=None, type='data-driven', time_domain=False,
                station_id=station.get_id(), channel_id=channel_id)

            channel.set_frequency_spectrum(
                channel.get_frequency_spectrum() + noise_spec, "same")

            if channel_id in self.deep_trigger_channels:
                trigger_noise_spec = noise_spec * get_response_conversion(det, station.get_id(), channel_id)
                trigger_channel = channel.get_trigger_channel()
                trigger_channel.set_frequency_spectrum(
                    trigger_channel.get_frequency_spectrum() + trigger_noise_spec, "same")

if __name__ == "__main__":

    ABS_PATH_HERE = str(os.path.dirname(os.path.realpath(__file__)))
    def_data_dir = os.path.join(ABS_PATH_HERE, "data")
    default_config_path = os.path.join(ABS_PATH_HERE, "../07_RNO_G_simulation/RNO_config.yaml")

    parser = argparse.ArgumentParser(description="Run NuRadioMC simulation")
    # Sim steering arguments
    parser.add_argument("--config", type=str, default=default_config_path, help="NuRadioMC yaml config file")
    parser.add_argument("--detectordescription", '--det', type=str, default=None, help="Path to RNO-G detector description file. If None, query from DB")
    parser.add_argument("--station_id", type=int, default=None, help="Set station to be used for simulation", required=True)

    # Neutrino arguments
    parser.add_argument("--energy", '-e', default=1e18, type=float, help="Neutrino energy [eV]")
    parser.add_argument("--flavor", '-f', default="all", type=str, help="the flavor")
    parser.add_argument("--interaction_type", '-it', default="ccnc", type=str, help="interaction type cc, nc or ccnc")

    # File meta-variables
    parser.add_argument("--index", '-i', default=0, type=int, help="counter to create a unique data-set identifier")
    parser.add_argument("--n_events_per_file", '-n', type=int, default=1e3, help="Number of nu-interactions per file")
    parser.add_argument("--data_dir", type=str, default=def_data_dir, help="directory name where the library will be created")
    parser.add_argument("--proposal", action="store_true", help="Use PROPOSAL for simulation")
    parser.add_argument("--nur_output", action="store_true", help="Write nur files.")

    args = parser.parse_args()
    kwargs = args.__dict__
    assert args.station_id is not None, "Please specify a station id with `--station_id`"

    # Defaults for the trigger simulation which are not yet in the hardware DB
    defaults = {
        "trigger_adc_sampling_frequency": 0.472,
        "trigger_adc_nbits": 8,
        "trigger_adc_noise_count": 5,
        "trigger_adc_min_voltage": -1,
        "trigger_adc_max_voltage": 1,
    }

    det = rnog_detector.Detector(
        detector_file=args.detectordescription, log_level=logging.INFO,
        always_query_entire_description=False, select_stations=args.station_id,
        over_write_handset_values=defaults)

    det.update(dt.datetime(2023, 8, 3))

    volume = get_fiducial_volume(args.energy)

    # Simulate fiducial volume around station
    pos = det.get_absolute_position(args.station_id)
    print(f"Simulating around center x0={pos[0]:.2f}m, y0={pos[1]:.2f}m")
    volume.update({"x0": pos[0], "y0": pos[1]})

    output_path = f"{args.data_dir}/station_{args.station_id}/nu_{args.flavor}_{args.interaction_type}"

    if not os.path.exists(output_path):
        print("Making dirs", output_path)
        os.makedirs(output_path, exist_ok=True)

    output_filename = f"{output_path}/{args.flavor}_{args.interaction_type}_1e{np.log10(args.energy):.2f}eV_{args.index:08d}.hdf5"

    flavor_ids = {"e": [12, -12], "mu": [14, -14], "tau": [16, -16], "all": [12, 14, 16, -12, -14, -16]}
    run_proposal = args.proposal and ("cc" in args.interaction_type) and (args.flavor in ["mu", "tau", "all"])
    if run_proposal:
        print(f"Using PROPOSAL for simulation of {args.flavor} {args.interaction_type}")

    input_data = generator.generate_eventlist_cylinder(
        "on-the-fly",
        kwargs["n_events_per_file"],
        args.energy, args.energy,
        volume,
        start_event_id=args.index * args.n_events_per_file + 1,
        flavor=flavor_ids[args.flavor],
        n_events_per_file=None,
        deposited=False,
        proposal=run_proposal,
        proposal_config="Greenland",
        start_file_id=0,
        log_level=None,
        proposal_kwargs={},
        max_n_events_batch=args.n_events_per_file,
        write_events=False,
        seed=root_seed + args.index,
        interaction_type=args.interaction_type,
    )

    if args.nur_output:
        nur_output_filename = output_filename.replace(".hdf5", ".nur")
    else:
        nur_output_filename = None

    sim = mySimulation(
        inputfilename=input_data,
        outputfilename=output_filename,
        det=det,
        evt_time=dt.datetime(2023, 8, 3),
        outputfilenameNuRadioReco=nur_output_filename,
        config_file=args.config,
        trigger_channels=deep_trigger_channels,
        # log_level=logging.INFO,
    )

    sim.run()