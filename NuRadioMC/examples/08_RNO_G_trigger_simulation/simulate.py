#!/bin/env python3
"""RNO-G neutrino/CR simulation with calibrated FLOWER trigger.

Supports two noise modes:
- Thermal noise (default): generated from temperature + signal chain response
- Measured FT noise (--ft_noise_dir): injected from real forced-trigger data

Both modes use the full FLOWER trigger model: hardware response,
triggerBoardResponse (VGA gain + ADC), and highLowThreshold with
calibrated thresholds.

Based on RNO_G_trigger_simulation/simulate.py with additions:
- Measured FT noise injection via noiseImporter (with trigger copy support)
- Asymmetric ADC saturation from pedestal voltage
- Hardware response padding for linear convolution
- Per-event ledger output
"""

import argparse
import numpy as np
import os
import secrets
import datetime as dt
import pandas as pd
import yaml

from scipy.fft import next_fast_len

from NuRadioMC.EvtGen import generator
from NuRadioMC.simulation import simulation
from NuRadioReco.utilities import units, signal_processing

from NuRadioReco.detector.RNO_G import rnog_detector

import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.efieldToVoltageConverterPerEfield
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.framework.sim_station

from NuRadioReco.modules.RNO_G import hardwareResponseIncorporator, triggerBoardResponse
from NuRadioReco.modules.trigger import highLowThreshold
from NuRadioReco.modules.measured_noise.RNO_G.noiseImporter import noiseImporter

import logging
logger = logging.getLogger("NuRadioMC.RNOG_trigger_simulation")
logger.setLevel(logging.INFO)

DEEP_TRIGGER_CHANNELS = [0, 1, 2, 3]
DEFAULT_PEDESTAL_V = 1.5

# Module-level state for the resampler monkey-patch
_noise_importer = None
_adc_clip_range = None


def RNO_G_HighLow_Thresh(lgRate_per_hz):
    """Threshold in sigma for a given trigger rate.

    Parameterization from the RNO-G hardware (IGLU + FLOWER LPF).
    """
    return (-859 + np.sqrt(39392706 - 3602500 * lgRate_per_hz)) / 1441.0


def get_vrms_from_temperature_for_trigger_channels(det, station_id, trigger_channels, temperature):
    """Compute trigger-path thermal Vrms per channel from temperature."""
    vrms_per_channel = []
    for channel_id in trigger_channels:
        resp = det.get_signal_chain_response(station_id, channel_id, trigger=True)
        vrms_per_channel.append(
            signal_processing.calculate_vrms_from_temperature(
                temperature=temperature, response=resp))
    return np.array(vrms_per_channel)


def get_fiducial_volume_neutrino(energy):
    """Energy-dependent fiducial volume for neutrino simulations."""
    max_radius_shallow = {
        16.25: 1.5, 16.5: 2.1, 16.75: 2.7, 17.0: 3.1, 17.25: 3.7,
        17.5: 3.9, 17.75: 4.4, 18.00: 4.8, 18.25: 5.1, 18.50: 5.25,
        18.75: 5.3, 19.0: 5.6, 100: 6.1,
    }
    min_z_shallow = {
        16.25: -0.65, 16.50: -0.8, 16.75: -1.2, 17.00: -1.5, 17.25: -1.7,
        17.50: -2.0, 17.75: -2.1, 18.00: -2.3, 18.25: -2.4, 18.50: -2.55,
        100: -2.7,
    }

    def get_limits(dic, E):
        idx = np.arange(len(dic))[E - 10 ** np.array(list(dic.keys())) * units.eV <= 0]
        assert len(idx), f"Energy {E} is too high."
        return np.array(list(dic.values()))[np.amin(idx)] * units.km

    return {
        "fiducial_rmax": get_limits(max_radius_shallow, energy),
        "fiducial_rmin": 0 * units.km,
        "fiducial_zmin": get_limits(min_z_shallow, energy),
        "fiducial_zmax": 0,
    }


def get_fiducial_volume_cr(rmax=200.0):
    """Shallow fiducial volume for CR proxy simulations (0-1m depth)."""
    return {
        "fiducial_rmax": rmax * units.m,
        "fiducial_rmin": 0,
        "fiducial_zmin": -1.0 * units.m,
        "fiducial_zmax": 0,
    }


if __name__ == "__main__":

    # Monkey-patch the resampler to add FT noise + ADC clipping.
    # Inside __main__ to avoid side effects on import.
    _original_resampler_run = simulation.channelResampler.run

    def resampler_with_noise_and_clip(event, station, detector, **kwargs):
        """Resample, optionally inject FT noise for readout, then clip."""
        _original_resampler_run(event, station, detector, **kwargs)

        if isinstance(station, NuRadioReco.framework.sim_station.SimStation):
            return

        # Inject FT noise into readout channels (stage 2)
        if _noise_importer is not None:
            _noise_importer.run(event, station, detector)
            logger.debug("Stage 2: readout noise injected")

        # ADC saturation clipping
        if _adc_clip_range is not None:
            lo, hi = _adc_clip_range
            for channel in station.iter_channels():
                trace = channel.get_trace()
                channel.set_trace(
                    np.clip(trace, lo, hi),
                    channel.get_sampling_rate())

    simulation.channelResampler.run = resampler_with_noise_and_clip

    parser = argparse.ArgumentParser(
        description="RNO-G simulation with calibrated FLOWER trigger")

    parser.add_argument("--config", type=str, default=None,
                        help="NuRadioMC YAML config file")
    parser.add_argument("--station_id", type=int, required=True)
    parser.add_argument("--detector_file", '--det', type=str, default=None,
                        help="Detector description file (default: query MongoDB)")

    # Event generation
    parser.add_argument("--neutrino_file", type=str, default=None,
                        help="Pre-generated HDF5 input file")
    parser.add_argument("--energy", '-e', default=1e18, type=float,
                        help="Neutrino energy in eV")
    parser.add_argument("--flavor", '-f', default="e", type=str,
                        choices=["e", "mu", "tau", "all"])
    parser.add_argument("--interaction_type", '-it', default="cc", type=str,
                        choices=["cc", "nc", "ccnc"])
    parser.add_argument("--n_events", '-n', type=int, default=1000)
    parser.add_argument("--fiducial_rmax", type=float, default=None,
                        help="Override config fiducial_volume.rmax (m)")
    parser.add_argument("--min_zenith", type=float, default=None,
                        help="Override config fiducial_volume.min_zenith (deg)")
    parser.add_argument("--max_zenith", type=float, default=None,
                        help="Override config fiducial_volume.max_zenith (deg)")

    # Output
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--index", '-i', default=0, type=int)
    parser.add_argument("--nur_output", action="store_true")

    # FT noise injection
    parser.add_argument("--ft_noise_dir", type=str, default=None,
                        help="FT noise data directory (enables measured noise mode)")
    parser.add_argument("--ft_seed", type=int, default=None)
    parser.add_argument("--ft_clean_mask", type=str, default=None)
    parser.add_argument("--trigger_vrms", type=str, default=None,
                        help="YAML file with trigger-path Vrms per channel "
                             "(from noise_analysis/trigger_vrms/extract_trigger_vrms.py)")

    # ADC pedestal
    parser.add_argument("--pedestal_voltage", type=float, default=DEFAULT_PEDESTAL_V,
                        help="ADC pedestal voltage in V (default: 1.5)")

    # Per-channel noise temperatures (workaround until DB has calibrated values)
    parser.add_argument("--noise_temperatures", type=str, default=None,
                        help="JSON file mapping channel_id to noise temperature (K). "
                             "Overrides the detector description per-channel values.")

    # Misc
    parser.add_argument("--proposal", action="store_true")
    parser.add_argument("--event_time", type=str, default="2022-10-01")

    args = parser.parse_args()

    # Determine noise mode
    use_ft_noise = args.ft_noise_dir is not None
    if use_ft_noise:
        logger.info(f"Using measured FT noise from {args.ft_noise_dir}")
    else:
        logger.info("Using thermal noise")

    # Config
    script_dir = os.path.dirname(os.path.realpath(__file__))
    if args.config is None:
        args.config = os.path.join(script_dir, "RNO_config.yaml")
    config = simulation.get_config(args.config)

    _override_noise_false = use_ft_noise and config.get("noise", True)

    # Detector
    det = rnog_detector.Detector(
        detector_file=args.detector_file, log_level=logging.INFO,
        always_query_entire_description=False,
        select_stations=args.station_id)

    event_time = dt.datetime.fromisoformat(args.event_time)
    det.update(event_time)

    # Override per-channel noise temperatures if provided.
    # Temporary workaround: the DB currently stores a flat 300 K default.
    # Once calibrated per-channel values are in the DB, this won't be needed.
    if args.noise_temperatures is not None:
        import json
        with open(args.noise_temperatures) as f:
            temp_map = json.load(f)
        for ch_id_str, temp_k in temp_map.items():
            det.get_channel(args.station_id, int(ch_id_str))["noise_temperature"] = float(temp_k)
        logger.info(f"Loaded per-channel noise temperatures from {args.noise_temperatures}")

    # ADC clip range from pedestal
    det_ch = det.get_channel(args.station_id, 0)
    adc_min = det_ch.get("adc_min_voltage", 0) * units.V
    adc_max = det_ch.get("adc_max_voltage", 2.5) * units.V
    _adc_clip_range = (adc_min - args.pedestal_voltage * units.V,
                       adc_max - args.pedestal_voltage * units.V)
    logger.info(f"ADC clip range (pedestal={args.pedestal_voltage:.2f}V): "
                f"[{_adc_clip_range[0]/units.mV:.0f}, {_adc_clip_range[1]/units.mV:.0f}] mV")

    # Trigger thresholds
    high_low_trigger_thresholds = {
        "1Hz": RNO_G_HighLow_Thresh(0),
    }

    # Trigger noise Vrms
    if use_ft_noise:
        if args.trigger_vrms is None:
            raise ValueError(
                "--trigger_vrms is required in FT noise mode. "
                "Generate it with noise_analysis/trigger_vrms/extract_trigger_vrms.py")
        with open(args.trigger_vrms) as f:
            vrms_data = yaml.safe_load(f)
        vrms_station = vrms_data.get("metadata", {}).get("station_id")
        if vrms_station is not None and vrms_station != args.station_id:
            logger.warning(f"Trigger Vrms file is for station {vrms_station}, "
                           f"but simulating station {args.station_id}")
        trigger_vrms_dict = vrms_data["trigger_vrms_V"]
        trigger_noise_vrms = np.array(
            [trigger_vrms_dict[ch] for ch in DEEP_TRIGGER_CHANNELS])
    else:
        if args.noise_temperatures is not None:
            # Per-channel temperatures were patched into the detector;
            # compute trigger Vrms from each channel's own temperature
            trigger_noise_vrms = np.array([
                get_vrms_from_temperature_for_trigger_channels(
                    det, args.station_id, [ch],
                    det.get_noise_temperature(args.station_id, ch))[0]
                for ch in DEEP_TRIGGER_CHANNELS])
        else:
            trigger_noise_vrms = get_vrms_from_temperature_for_trigger_channels(
                det, args.station_id, DEEP_TRIGGER_CHANNELS,
                config['trigger']['noise_temperature'])

    logger.info(f"Trigger Vrms: {[f'{v/units.mV:.2f} mV' for v in trigger_noise_vrms]}")

    # Initialize modules
    hw_resp = hardwareResponseIncorporator.hardwareResponseIncorporator()
    hw_resp.begin(trigger_channels=DEEP_TRIGGER_CHANNELS)

    adc_resp = triggerBoardResponse.triggerBoardResponse()
    adc_resp.begin(clock_offset=0.0, adc_output="counts")

    trigger_sim = highLowThreshold.triggerSimulator()

    # Configure the framework's singleton efieldToVoltageConverter with
    # padding for linear convolution (prevents circular wrap artifacts)
    from NuRadioMC.simulation.simulation import efieldToVoltageConverter
    efieldToVoltageConverter.begin(
        caching=False,
        pre_pulse_time=400 * units.ns,
        post_pulse_time=2000 * units.ns,
    )

    # FT noise importer (if using measured noise)
    if use_ft_noise:
        # Discover FT noise files explicitly (avoid recursive glob issues)
        import glob as _glob
        import uproot as _uproot
        ft_dir = args.ft_noise_dir
        ft_files = sorted(_glob.glob(os.path.join(ft_dir, "station*_run*.root")))
        if not ft_files:
            ft_files = sorted(_glob.glob(os.path.join(ft_dir, "run*/waveforms.root")))
        if not ft_files:
            raise FileNotFoundError(f"No FT ROOT files found in {ft_dir}")

        # Validate files (skip corrupt ones that crash mattak)
        valid_files = []
        for f in ft_files:
            try:
                with _uproot.open(f) as rf:
                    if "combined" in rf or "waveforms" in rf:
                        valid_files.append(f)
            except Exception:
                pass
        if len(valid_files) < len(ft_files):
            logger.warning(f"Skipped {len(ft_files) - len(valid_files)} corrupt FT files")
        ft_files = valid_files
        logger.info(f"Found {len(ft_files)} valid FT noise files")

        # Build event selectors (FORCE trigger + optional clean mask)
        ft_selectors = [lambda einfo: einfo.triggerType == "FORCE"]
        if args.ft_clean_mask is not None:
            mask_data = np.load(args.ft_clean_mask)
            if 'station_id' in mask_data:
                mask_station = int(mask_data['station_id'])
                if mask_station != args.station_id:
                    logger.warning(f"Clean mask is for station {mask_station}, "
                                   f"but simulating station {args.station_id}")
            flagged = set()
            for r, e, c in zip(mask_data['runNum'], mask_data['eventNum'],
                               mask_data['is_clean']):
                if c == 0:
                    flagged.add((int(r), int(e)))
            ft_selectors.append(
                lambda einfo, _f=flagged: (einfo.run, einfo.eventNumber) not in _f)
            logger.info(f"Clean mask: excluding {len(flagged)} flagged FT events")

        _noise_importer_instance = noiseImporter()
        _noise_importer_instance.begin(
            noise_files=ft_files,
            match_station_id=True,
            scramble_noise_file_order=True,
            random_seed=args.ft_seed,
            inject_trigger_copies=True,
            trigger_channels=DEEP_TRIGGER_CHANNELS,
            hardware_response_incorporator=hw_resp,
            reader_kwargs={
                "selectors": ft_selectors,
                "select_runs": False,
                "convert_to_voltage": True,
                "apply_baseline_correction": "median",
            },
        )
        _noise_importer = _noise_importer_instance

        n_pool = _noise_importer.n_events_available
        if args.n_events > n_pool:
            logger.warning(
                f"FT noise pool ({n_pool} events) is smaller than n_events "
                f"({args.n_events}). Noise events will be reused. Consider "
                f"adding more FT data to reduce repetition.")

    # Fiducial volume + zenith range: CLI overrides config, config overrides defaults
    fid_config = config.get("fiducial_volume", {})
    fiducial_rmax = args.fiducial_rmax if args.fiducial_rmax is not None else fid_config.get("rmax")
    fiducial_zmin = fid_config.get("zmin")
    fiducial_zmax = fid_config.get("zmax")

    min_zenith = args.min_zenith if args.min_zenith is not None else fid_config.get("min_zenith", 0.0)
    max_zenith = args.max_zenith if args.max_zenith is not None else fid_config.get("max_zenith", 60.0)

    if fiducial_rmax is not None and fiducial_zmin is not None:
        volume = {
            "fiducial_rmax": fiducial_rmax * units.m,
            "fiducial_rmin": 0,
            "fiducial_zmin": fiducial_zmin * units.m,
            "fiducial_zmax": (fiducial_zmax or 0) * units.m,
        }
        logger.info(f"Fiducial volume: rmax={fiducial_rmax}m, "
                     f"z=[{fiducial_zmin}, {fiducial_zmax or 0}]m")
    elif fiducial_rmax is not None:
        volume = get_fiducial_volume_cr(rmax=fiducial_rmax)
        logger.info(f"Fiducial volume: rmax={fiducial_rmax}m, z=[-1, 0]m")
    else:
        volume = get_fiducial_volume_neutrino(args.energy)

    logger.info(f"Zenith range: [{min_zenith}, {max_zenith}] deg")

    pos = det.get_absolute_position(args.station_id)
    logger.info(f"Simulating around center x0={pos[0]:.2f}m, y0={pos[1]:.2f}m")
    volume.update({"x0": pos[0], "y0": pos[1]})

    # Simulation class
    class mySimulation(simulation.simulation):
        """Simulation subclass with FLOWER trigger and optional FT noise."""

        def __init__(self, *args_init, **kwargs_init):
            if not use_ft_noise:
                tmp_config = simulation.get_config(kwargs_init["config_file"])
                noise_temp = tmp_config['trigger']['noise_temperature']

                # When noise_temperature is "detector", the framework handles
                # per-channel noise itself; don't override with a flat Vrms.
                if noise_temp == "detector":
                    def wrapper_detector_simulation(*a, **kw):
                        kw['add_noise'] = False
                        detector_simulation_thermal(*a, **kw)
                else:
                    def wrapper_detector_simulation(*a, **kw):
                        noise_vrms = signal_processing.calculate_vrms_from_temperature(
                            temperature=noise_temp,
                            bandwidth=tmp_config["sampling_rate"] / 2)
                        kw['noise_vrms'] = noise_vrms
                        kw['max_freq'] = tmp_config["sampling_rate"] / 2
                        detector_simulation_thermal(*a, **kw)

                self._detector_simulation_part2 = wrapper_detector_simulation

            super().__init__(*args_init, **kwargs_init)

            from NuRadioMC.simulation import simulation as sim_module
            sim_module.efieldToVoltageConverterPerEfield.begin(
                pre_pulse_time=400 * units.ns,
                post_pulse_time=2000 * units.ns,
            )

            self.event_log = []
            self._readout_to_trigger_transfer = {}

        def _detector_simulation_filter_amp(self, evt, station, det_arg):
            """Apply hardware response with padding, then inject FT trigger noise."""
            is_sim = isinstance(station, NuRadioReco.framework.sim_station.SimStation)

            # Pad non-trigger channels for linear convolution
            _hw_pad_info = []
            for channel in station.iter_channels():
                ch_id = channel.get_id()
                if not is_sim and ch_id in DEEP_TRIGGER_CHANNELS:
                    continue
                trace = channel.get_trace()
                N = len(trace)
                sr = channel.get_sampling_rate()
                n_pad = int(np.ceil(2000 * sr))
                Npad = next_fast_len(N + n_pad)
                padded = np.zeros(Npad, dtype=trace.dtype)
                padded[:N] = trace
                channel.set_trace(padded, sr)
                _hw_pad_info.append((channel, N))

            hw_resp.run(evt, station, det_arg, sim_to_data=True)

            for channel, N in _hw_pad_info:
                channel.set_trace(
                    channel.get_trace()[:N],
                    channel.get_sampling_rate())

            if is_sim:
                return

            # Stage 1: inject FT noise into trigger copies only (at 5 GHz).
            if _noise_importer is not None:
                _noise_importer.run(evt, station, det_arg, trigger_copies_only=True)
                logger.debug("Stage 1: trigger copy noise injected")

        def _detector_simulation_trigger(self, evt, station, det_arg):
            """Run FLOWER trigger (triggerBoardResponse + highLowThreshold) and log results."""
            max_amps = {}
            for ch_id in DEEP_TRIGGER_CHANNELS:
                if station.has_channel(ch_id):
                    trace = station.get_channel(ch_id).get_trace()
                    max_amps[ch_id] = np.max(np.abs(trace))

            vrms_after_gain = adc_resp.run(
                evt, station, det_arg,
                trigger_channels=DEEP_TRIGGER_CHANNELS,
                vrms=trigger_noise_vrms, digitize_trace=True)

            flower_rate = station.get_trigger_channel(
                DEEP_TRIGGER_CHANNELS[0]).get_sampling_rate()

            for thresh_key, threshold in high_low_trigger_thresholds.items():
                threshold_high = {ch: int(round(threshold * vrms))
                                  for ch, vrms in zip(DEEP_TRIGGER_CHANNELS, vrms_after_gain)}
                threshold_low = {ch: int(round(-threshold * vrms))
                                 for ch, vrms in zip(DEEP_TRIGGER_CHANNELS, vrms_after_gain)}

                trigger_sim.run(
                    evt, station, det_arg,
                    threshold_high=threshold_high,
                    threshold_low=threshold_low,
                    use_digitization=False,
                    high_low_window=6 / flower_rate,
                    coinc_window=20 / flower_rate,
                    number_concidences=2,
                    triggered_channels=DEEP_TRIGGER_CHANNELS,
                    trigger_name=f"deep_high_low_{thresh_key}",
                    pre_trigger_time=200 * units.ns,
                )

            row = {
                'event_group_id': evt.get_run_number(),
                'event_id': evt.get_id(),
                'triggered': station.has_triggered(),
            }
            for ch_id in DEEP_TRIGGER_CHANNELS:
                row[f'max_amp_ch{ch_id}_mV'] = max_amps.get(ch_id, np.nan) / units.mV
            self.event_log.append(row)

    _noise_adder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()

    def detector_simulation_thermal(evt, station, det_arg, noise_vrms=None,
                                     max_freq=None, add_noise=True):
        """Thermal noise detector simulation (no FT noise)."""
        efieldToVoltageConverter.run(evt, station, det_arg,
                                     channel_ids=DEEP_TRIGGER_CHANNELS)
        if add_noise and noise_vrms is not None:
            _noise_adder.run(
                evt, station, det_arg, amplitude=noise_vrms,
                min_freq=0 * units.MHz, max_freq=max_freq, type='rayleigh')
        hw_resp.run(evt, station, det_arg, sim_to_data=True)

    # Event generation
    root_seed = secrets.randbits(128)
    flavor_ids = {"e": [12, -12], "mu": [14, -14], "tau": [16, -16],
                  "all": [12, 14, 16, -12, -14, -16]}

    if args.neutrino_file is None:
        zen_min = np.deg2rad(min_zenith)
        zen_max = np.deg2rad(max_zenith)

        input_data = generator.generate_eventlist_cylinder(
            "on-the-fly",
            args.n_events,
            args.energy, args.energy,
            volume,
            thetamin=zen_min, thetamax=zen_max,
            start_event_id=args.index * args.n_events + 1,
            flavor=flavor_ids[args.flavor],
            n_events_per_file=None,
            deposited=False,
            proposal=args.proposal,
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

    # Output paths
    if args.output_file:
        output_hdf5 = os.path.join(args.data_dir, args.output_file)
    else:
        output_hdf5 = os.path.join(
            args.data_dir,
            f"{args.flavor}_{args.interaction_type}"
            f"_1e{np.log10(args.energy):.2f}eV_{args.index:08d}.hdf5")

    os.makedirs(args.data_dir, exist_ok=True)
    output_nur = output_hdf5.replace(".hdf5", ".nur") if args.nur_output else None

    # Run simulation
    sim = mySimulation(
        inputfilename=input_data,
        outputfilename=output_hdf5,
        det=det,
        evt_time=event_time,
        outputfilenameNuRadioReco=output_nur,
        config_file=args.config,
        trigger_channels=DEEP_TRIGGER_CHANNELS,
        file_overwrite=True,
    )

    if _override_noise_false:
        logger.warning("FT noise mode: setting noise=False to prevent "
                       "thermal noise being added on top of injected FT noise")
        sim._config['noise'] = False

    n_triggered = sim.run()

    # Build full event ledger (all input events, including efield_cut)
    fin = sim._fin
    input_egids = np.unique(fin['event_group_ids'])

    trigger_log = pd.DataFrame(sim.event_log)
    reached_trigger = set()
    if len(trigger_log):
        reached_trigger = set(trigger_log['event_group_id'].values)

    rows = []
    for egid in input_egids:
        idx = np.where(fin['event_group_ids'] == egid)[0][0]
        row = {
            'event_group_id': int(egid),
            'zenith_deg': np.rad2deg(fin['zeniths'][idx]),
            'azimuth_deg': np.rad2deg(fin['azimuths'][idx]),
            'energy_eV': fin['energies'][idx],
            'flavor': int(fin['flavors'][idx]),
        }

        if egid in reached_trigger:
            evt_rows = trigger_log[trigger_log['event_group_id'] == egid]
            if evt_rows['triggered'].any():
                row['status'] = 'triggered'
            else:
                row['status'] = 'trigger_failed'
            for ch_id in DEEP_TRIGGER_CHANNELS:
                col = f'max_amp_ch{ch_id}_mV'
                row[col] = evt_rows[col].max()
        else:
            row['status'] = 'efield_cut'
            for ch_id in DEEP_TRIGGER_CHANNELS:
                row[f'max_amp_ch{ch_id}_mV'] = np.nan

        rows.append(row)

    ledger = pd.DataFrame(rows)
    ledger_path = output_hdf5.replace('.hdf5', '_ledger.csv')
    ledger.to_csv(ledger_path, index=False)
    logger.info(f"Ledger: {ledger_path} ({n_triggered} triggered / {len(ledger)} total)")
