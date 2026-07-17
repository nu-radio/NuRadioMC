#!/bin/env python3
"""RNO-G neutrino/CR simulation with calibrated FLOWER trigger.

Supports two noise modes:
- Thermal noise (default): generated from temperature + signal chain response
- Measured FT noise (--ft_noise_dir): injected from real forced-trigger data

Both modes use the full FLOWER trigger model: hardware response,
triggerBoardResponse (VGA gain + ADC), and highLowThreshold with
calibrated thresholds.

Measured FT noise uses the v9 production method (matching
08_RNO_G_trigger_simulation_testing/.../simulate_fixed_response_v9.py):
- Trigger channel copies (ch 0-3) get FT noise at the 5 GHz internal rate,
  built by upsampling FT tiles and stitching them with a Hann overlap-add to
  span the full internal trace length, then transformed readout->trigger via
  the hardware-response ratio.
- Readout channels get a separate FT realization added at the native 3.2 GHz
  rate after the readout-window cut and resample.
- The readout window is cut with a zero-padded cutter instead of the stock
  cyclic roll, so overflow past the trace edge is filled with zeros (later
  covered by the readout FT injection) rather than wrapped.

Based on RNO_G_trigger_simulation/simulate.py with additions:
- Measured FT noise injection (in-script tiled pool, trigger + readout paths)
- Asymmetric ADC saturation from pedestal voltage
- Hardware response padding for linear convolution
- Per-event ledger output
"""

import argparse
import numpy as np
import os
import secrets
import datetime as dt
from collections import deque
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
from NuRadioReco.modules.channelReadoutWindowCutter import _get_number_of_samples

import logging
logger = logging.getLogger("NuRadioMC.RNOG_trigger_simulation")
logger.setLevel(logging.INFO)

DEEP_TRIGGER_CHANNELS = [0, 1, 2, 3]
DEFAULT_PEDESTAL_V = 1.5
TILE_OVERLAP = 200  # samples of Hann crossfade between FT tiles at 5 GHz

# Module-level state for the monkey-patched resampler/cutter
_ft_noise_pool = None
_adc_clip_range = None
_adc_clip_per_channel = None
# st13 measured ch0 trigger model ("normal" | "measured_8x" | "measured_dead")
_ch0_trigger_model = "normal"


class FTNoisePool:
    """Streaming pool of forced-trigger noise traces from ROOT files.

    Loads one ROOT file at a time (~560 events, ~210 MB) and hands out one
    event's traces per call. Events flagged as non-thermal by the clean mask
    are skipped. Files are cycled and reshuffled indefinitely so a pool
    smaller than the number of thrown events simply reuses realizations.
    """

    def __init__(self, ft_dir, station_id=23, seed=None, clean_mask_path=None):
        """Discover FT ROOT files and load the contamination mask.

        Args:
            ft_dir: Directory of ``station{station_id}_run*.root`` files.
            station_id: Station whose channels to read.
            seed: RNG seed for file shuffling (reproducible event order).
            clean_mask_path: Optional NPZ with ``runNum``/``eventNum``/
                ``is_clean``; events with ``is_clean == 0`` are skipped.

        Raises:
            FileNotFoundError: If no matching ROOT files are found.
        """
        self._station_id = station_id
        self._rng = np.random.default_rng(seed)
        self._buffer = deque()
        self._files_loaded = 0

        self._ft_files = sorted([
            os.path.join(ft_dir, f)
            for f in os.listdir(ft_dir)
            if f.startswith(f"station{station_id}_run") and f.endswith(".root")
        ])
        self._rng.shuffle(self._ft_files)

        if not self._ft_files:
            raise FileNotFoundError(
                f"No FT ROOT files matching station{station_id}_run*.root in {ft_dir}")

        self._file_idx = 0

        self._flagged = set()
        if clean_mask_path:
            if not os.path.exists(clean_mask_path):
                raise FileNotFoundError(
                    f"FT clean mask not found: {clean_mask_path}. Running "
                    f"unmasked injects contaminated FT events that inflate "
                    f"the noise-trigger rate; omit the mask argument only "
                    f"deliberately.")
            mask_data = np.load(clean_mask_path)
            for r, e, c in zip(mask_data['runNum'], mask_data['eventNum'],
                               mask_data['is_clean']):
                if c == 0:
                    self._flagged.add((int(r), int(e)))
            logger.info(f"FTNoisePool: loaded clean mask, "
                        f"{len(self._flagged)} flagged events")

        logger.info(f"FTNoisePool: {len(self._ft_files)} ROOT files in {ft_dir}")

    def _load_next_file(self):
        """Read the next ROOT file's FORCE events into the buffer.

        Skips corrupt files and reshuffles when the file list is exhausted.

        Raises:
            RuntimeError: If no events load after several file attempts.
        """
        from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData

        max_retries = 10
        for _ in range(max_retries):
            if self._file_idx >= len(self._ft_files):
                self._file_idx = 0
                self._rng.shuffle(self._ft_files)
                logger.info("FTNoisePool: cycled through all files, reshuffling")

            fpath = self._ft_files[self._file_idx]
            self._file_idx += 1
            self._files_loaded += 1

            try:
                reader = readRNOGData()
                reader.begin(
                    [fpath],
                    convert_to_voltage=True,
                    apply_baseline_correction="median",
                    selectors=[lambda einfo: einfo.triggerType == "FORCE"],
                    select_runs=False,
                    read_calibrated_data=False,
                )
            except Exception as e:
                logger.warning(f"FTNoisePool: skipping corrupt file "
                               f"{os.path.basename(fpath)}: {e}")
                continue

            loaded = 0
            skipped = 0
            try:
                for evt in reader.run():
                    station = evt.get_station(self._station_id)
                    if station is None:
                        continue

                    key = (evt.get_run_number(), evt.get_id())
                    if key in self._flagged:
                        skipped += 1
                        continue

                    traces = {ch.get_id(): ch.get_trace().copy()
                              for ch in station.iter_channels()}
                    if traces:
                        self._buffer.append(traces)
                        loaded += 1
            except Exception as e:
                logger.warning(f"FTNoisePool: error reading events from "
                               f"{os.path.basename(fpath)}: {e}")

            try:
                reader.end()
            except Exception:
                pass

            if self._files_loaded <= 3 or self._files_loaded % 50 == 0:
                logger.info(f"FTNoisePool: loaded {loaded} events from "
                            f"{os.path.basename(fpath)} (skipped {skipped} flagged)")

            if loaded > 0:
                return

        raise RuntimeError(
            f"FTNoisePool: failed to load events after {max_retries} files")

    def get_noise_event(self):
        """Return one event's traces as ``{channel_id: np.array}``.

        Traces are in Volts, 2048 samples at 3.2 GHz. Loads the next file
        when the buffer empties.
        """
        if not self._buffer:
            self._load_next_file()
        return self._buffer.popleft()


def upsample_trace(trace, target_n_samples):
    """Upsample a trace to ``target_n_samples`` via FFT zero-padding above Nyquist."""
    n_orig = len(trace)
    spec = np.fft.rfft(trace)
    new_spec = np.zeros(target_n_samples // 2 + 1, dtype=complex)
    new_spec[:len(spec)] = spec
    return np.fft.irfft(new_spec, n=target_n_samples) * (target_n_samples / n_orig)


def tile_noise_overlap_add(tiles, target_length, overlap=TILE_OVERLAP):
    """Stitch upsampled FT noise tiles into one trace with an equal-power crossfade.

    Interior seams use head weight sqrt(ramp) and tail weight sqrt(1 - ramp) so that
    head^2 + tail^2 = 1 at every overlap sample. For independent tiles the summed
    variance is then constant through the seam, unlike the linear Hann crossfade,
    which attenuates it. The first tile's leading edge and the last tile's trailing
    edge are left at full amplitude (no neighbor to fill them).

    Args:
        tiles: List of equal-length 1D arrays (upsampled FT traces at 5 GHz).
        target_length: Output length in samples.
        overlap: Crossfade width in samples.

    Returns:
        1D array of length ``target_length``.
    """
    if not tiles:
        return np.zeros(target_length)

    n_tile = len(tiles[0])
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(overlap) / overlap))
    head = np.sqrt(ramp)
    tail = np.sqrt(1.0 - ramp)

    total_len = n_tile + (len(tiles) - 1) * (n_tile - overlap)
    result = np.zeros(max(total_len, target_length + overlap))

    last = len(tiles) - 1
    pos = 0
    for k, tile in enumerate(tiles):
        windowed = tile.copy()
        if k > 0:
            windowed[:overlap] *= head
        if k < last:
            windowed[-overlap:] *= tail
        result[pos:pos + n_tile] += windowed
        pos += n_tile - overlap

    return result[:target_length]


def zero_padded_readout_window_cutter(event, station, detector):
    """Cut each channel to the readout window, zero-filling any overflow.

    Drop-in replacement for ``channelReadoutWindowCutter.run``. Where the
    readout window extends past the internal trace boundaries, the missing
    region is filled with zeros instead of being wrapped cyclically. In FT
    mode the readout FT-noise injection later covers those zeros with a real
    noise realization.

    Side effects:
        Replaces each channel's trace and trace start time in place.
    """
    trigger = station.get_primary_trigger()
    if trigger is None:
        trigger = station.get_first_trigger()
        if trigger is not None:
            trigger.set_primary(True)

    if trigger is None or not trigger.has_triggered():
        return

    trigger_time = trigger.get_trigger_time()

    for channel in station.iter_channels():
        channel_id = channel.get_id()
        sampling_rate = channel.get_sampling_rate()
        detector_sampling_rate = detector.get_sampling_frequency(
            station.get_id(), channel_id)
        detector_n_samples = detector.get_number_of_samples(
            station.get_id(), channel_id)

        number_of_samples, _ = _get_number_of_samples(
            sampling_rate, detector_sampling_rate, detector_n_samples,
            issue_error=True)

        trace = channel.get_trace()
        if number_of_samples > trace.shape[0]:
            raise AttributeError(
                f"Input has fewer samples ({trace.shape[0]}) "
                f"than desired output ({number_of_samples}).")

        pre_trigger_time = trigger.get_pre_trigger_time_channel(channel_id)
        readout_start_time = trigger_time - pre_trigger_time
        offset_time = readout_start_time - channel.get_trace_start_time()
        start_sample = int(round(offset_time * sampling_rate))
        end_sample = start_sample + number_of_samples

        result = np.zeros(number_of_samples)
        src_start = max(0, start_sample)
        src_end = min(len(trace), end_sample)
        if src_start < src_end:
            dst_start = src_start - start_sample
            result[dst_start:dst_start + (src_end - src_start)] = trace[src_start:src_end]

        channel.set_trace(result, sampling_rate)
        channel.set_trace_start_time(readout_start_time)


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
        """Look up the fiducial limit for a given energy from a threshold dict."""
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

        # Readout path: add a fresh FT realization directly at the native
        # 3.2 GHz rate (the noise was recorded through the readout chain, so
        # no readout->trigger transform is needed here). Independent from the
        # tiles used for the trigger copies.
        if _ft_noise_pool is not None:
            ft_evt = _ft_noise_pool.get_noise_event()
            for channel in station.iter_channels():
                ft_ch = ft_evt.get(channel.get_id())
                if ft_ch is not None and len(channel.get_trace()) == len(ft_ch):
                    channel.set_trace(
                        channel.get_trace() + ft_ch,
                        channel.get_sampling_rate())
            logger.debug("Stage 2: readout FT noise injected")

        # ADC saturation clipping. Per-channel asymmetric bounds from measured
        # pedestals when a clip_thresholds file is loaded; otherwise the uniform
        # range from the scalar pedestal_voltage.
        if _adc_clip_per_channel is not None or _adc_clip_range is not None:
            for channel in station.iter_channels():
                lo, hi = _adc_clip_range
                if _adc_clip_per_channel is not None:
                    lo, hi = _adc_clip_per_channel.get(channel.get_id(), (lo, hi))
                channel.set_trace(
                    np.clip(channel.get_trace(), lo, hi),
                    channel.get_sampling_rate())

    simulation.channelResampler.run = resampler_with_noise_and_clip

    parser = argparse.ArgumentParser(
        description="RNO-G simulation with calibrated FLOWER trigger")

    parser.add_argument("--config", type=str, default=None,
                        help="NuRadioMC YAML config file")
    parser.add_argument("--station_id", type=int, required=True)
    parser.add_argument("--detector_file", '--det', type=str, default=None,
                        help="Detector description file (default: RNOG_DETECTOR_FILE env var, "
                             "else query MongoDB)")
    parser.add_argument("--ch0_trigger_model",
                        choices=["normal", "measured_8x", "measured_dead"], default="normal",
                        help="st13 measured ch0 trigger model. measured_8x: ch0 count trace "
                             "x1/8 + 4-count absolute floor (>=8x-suppressed trigger path from "
                             "the daqstatus scaler bound). measured_dead: ch0 removed from the "
                             "2-of-4 (conservative bracket). normal: standard 3.759-sigma ch0.")

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
                        help="ADC pedestal voltage in V (default: 1.5); uniform-clip fallback "
                             "when --clip_thresholds is not given")
    parser.add_argument("--clip_thresholds", type=str, default=None,
                        help="YAML of per-channel ADC clip bounds {ch: [lo_mV, hi_mV]} "
                             "(from pedestal_extraction/pedestal_analysis.py); per-channel "
                             "asymmetric clip, overrides the uniform --pedestal_voltage clip")

    # Per-channel noise temperatures (workaround until DB has calibrated values)
    parser.add_argument("--noise_temperatures", type=str, default=None,
                        help="JSON file mapping channel_id to noise temperature (K). "
                             "Overrides the detector description per-channel values.")

    # Misc
    parser.add_argument("--proposal", action="store_true")
    parser.add_argument("--event_time", type=str, default="2022-10-01")

    args = parser.parse_args()
    _ch0_trigger_model = args.ch0_trigger_model

    # Determine noise mode
    use_ft_noise = args.ft_noise_dir is not None
    if use_ft_noise:
        logger.info(f"Using measured FT noise from {args.ft_noise_dir}")
        # Zero-pad readout-window overflow instead of the stock cyclic roll.
        # Only in FT mode: the post-cut readout FT injection refills the
        # zero region, whereas thermal mode has no post-cut noise stage and
        # relies on the stock behavior.
        simulation.channelReadoutWindowCutter.run = zero_padded_readout_window_cutter
    else:
        logger.info("Using thermal noise")

    # Config
    script_dir = os.path.dirname(os.path.realpath(__file__))
    if args.config is None:
        args.config = os.path.join(script_dir, "RNO_config.yaml")
    config = simulation.get_config(args.config)

    _override_noise_false = use_ft_noise and config.get("noise", True)

    # Detector. The file path may come from --detector_file or, so the production config
    # need not embed a detector-export path, from the RNOG_DETECTOR_FILE env var. If neither
    # is set, the DB is queried at --event_time.
    detector_file = args.detector_file or os.environ.get("RNOG_DETECTOR_FILE")
    det = rnog_detector.Detector(
        detector_file=detector_file, log_level=logging.INFO,
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

    if args.clip_thresholds is not None:
        with open(args.clip_thresholds) as f:
            clip_data = yaml.safe_load(f)
        _adc_clip_per_channel = {int(ch): (lo * units.mV, hi * units.mV)
                                 for ch, (lo, hi) in clip_data["clip_thresholds_mV"].items()}
        logger.info(f"Loaded per-channel ADC clip thresholds from {args.clip_thresholds} "
                    f"(ch0 [{_adc_clip_per_channel[0][0]/units.mV:.0f}, "
                    f"{_adc_clip_per_channel[0][1]/units.mV:.0f}] mV)")

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

    # FT noise pool (if using measured noise). The pool streams FORCE events
    # from station{id}_run*.root, skips clean-mask-flagged and corrupt events,
    # and cycles the file list so a pool smaller than n_events reuses
    # realizations. The readout->trigger transform is applied in-script (see
    # mySimulation), so no hardware-response handle is needed here.
    if use_ft_noise:
        _ft_noise_pool = FTNoisePool(
            ft_dir=args.ft_noise_dir,
            station_id=args.station_id,
            seed=args.ft_seed,
            clean_mask_path=args.ft_clean_mask,
        )

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

    class mySimulation(simulation.simulation):
        """Simulation subclass with FLOWER trigger and optional FT noise."""

        def __init__(self, *args_init, **kwargs_init):
            """Set up noise wrapper and configure efield converter padding."""
            # Set before super().__init__(): the parent constructor runs
            # _detector_simulation_filter_amp on a dummy event (noise-level
            # normalization), which for FT mode already reaches the trigger-copy
            # injection and the transfer cache.
            self.event_log = []
            self._readout_to_trigger_transfer = {}

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

            if is_sim or _ft_noise_pool is None:
                return

            # Stage 1: inject FT noise into the trigger copies (ch 0-3) at the
            # 5 GHz internal rate. Upsample FT tiles and Hann overlap-add them
            # to span the full internal trace, then transform readout->trigger
            # via the hardware-response ratio.
            n_internal = len(next(station.iter_channels()).get_trace())
            n_up = int(round(2048 * (5.0 / 3.2)))  # one FT event upsampled to 5 GHz
            stride = n_up - TILE_OVERLAP
            n_tiles = max(1, int(np.ceil(n_internal / stride)))
            ft_events = [_ft_noise_pool.get_noise_event() for _ in range(n_tiles)]

            for channel in station.iter_channels():
                ch_id = channel.get_id()
                if (ch_id not in DEEP_TRIGGER_CHANNELS
                        or not channel.has_extra_trigger_channel()):
                    continue

                tiles = [upsample_trace(ft_evt[ch_id], n_up)
                         for ft_evt in ft_events if ch_id in ft_evt]
                if not tiles:
                    continue

                noise = tile_noise_overlap_add(tiles, n_internal)
                transfer = self._get_readout_to_trigger_transfer(
                    ch_id, n_internal, det_arg, station.get_id())
                trig_noise = np.fft.irfft(np.fft.rfft(noise) * transfer, n=n_internal)

                trig_ch = channel.get_trigger_channel()
                trig_trace = trig_ch.get_trace()
                n_trig = len(trig_trace)
                trig_ch.set_trace(trig_trace + trig_noise[:n_trig],
                                  trig_ch.get_sampling_rate())
            logger.debug("Stage 1: trigger copy FT noise injected")

        def _get_readout_to_trigger_transfer(self, ch_id, n_samples, det_arg, station_id):
            """Return the cached readout->trigger transfer (trigger/readout filter ratio).

            FT noise carries the readout signal chain; the trigger copies need
            trigger-path noise. The ratio is evaluated on an ``n_samples`` rfft
            grid at the 5 GHz internal rate and regularized where the readout
            response is near zero.
            """
            key = (ch_id, n_samples)
            if key not in self._readout_to_trigger_transfer:
                ff = np.fft.rfftfreq(n_samples, d=1.0 / (5.0 * units.GHz))
                readout = hw_resp.get_filter(
                    ff, station_id, ch_id, det_arg, sim_to_data=True, is_trigger=False)
                trigger = hw_resp.get_filter(
                    ff, station_id, ch_id, det_arg, sim_to_data=True, is_trigger=True)
                readout_abs = np.abs(readout)
                max_r = np.max(readout_abs)
                safe_readout = np.where(readout_abs > 1e-3 * max_r, readout, max_r)
                self._readout_to_trigger_transfer[key] = trigger / safe_readout
            return self._readout_to_trigger_transfer[key]

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

                if _ch0_trigger_model != "normal":
                    # Measured st13 ch0 trigger model (daqstatus servo + Task B harness): ch0
                    # runs at an absolute 4-count floor on an >=8x-suppressed trigger path.
                    # Cast the trigger count traces to float so the fractional 1/8-scaled ch0
                    # and float thresholds satisfy get_high_low_triggers' dtype==type check
                    # (int-trace/int-threshold gives identical crossings, so ch1-3 are
                    # unchanged). ch1-3 stay at their 3.759-sigma count thresholds.
                    for _ch in DEEP_TRIGGER_CHANNELS:
                        _tc = station.get_trigger_channel(_ch)
                        _tc.set_trace(_tc.get_trace().astype(np.float64), _tc.get_sampling_rate())
                    threshold_high = {ch: float(v) for ch, v in threshold_high.items()}
                    threshold_low = {ch: float(v) for ch, v in threshold_low.items()}
                    if _ch0_trigger_model == "measured_8x":
                        _t0 = station.get_trigger_channel(0)
                        _t0.set_trace(_t0.get_trace() / 8.0, _t0.get_sampling_rate())
                        threshold_high[0] = 4.0
                        threshold_low[0] = -4.0
                    elif _ch0_trigger_model == "measured_dead":
                        threshold_high[0] = 1.0e9
                        threshold_low[0] = -1.0e9

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
