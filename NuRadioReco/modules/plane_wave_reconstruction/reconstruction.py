import itertools
import json
import logging
import lzma
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.signal
from astropy.time import Time

from joblib import Memory
from NuRadioMC.utilities import medium, medium_base
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.modules.channelSignalReconstructor import (
    channelSignalReconstructor,
)
from NuRadioReco.utilities.signal_processing import (
    resample, butterworth_filter_trace
)
from NuRadioReco.modules.RNO_G.dataProviderRNOG import dataProviderRNOG
from NuRadioReco.utilities import units, trace_utilities
from tqdm.auto import tqdm as tqdm_

from plotting import plot_flight_trajectories, plot_skymap
from travel_time_maps import load_map, load_map_non_interp

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

INBOX_RNOG_DATA = "/pnfs/ifh.de/acs/radio/diskonly/data/inbox/"
RNOG_DATA = "/pnfs/ifh.de/acs/radio/diskonly/data/full/root/"
NEW_DATA = "/pnfs/ifh.de/acs/radio/diskonly/new_data/"

logging.getLogger("NuRadioMC").setLevel(logging.ERROR)
logging.getLogger("NuRadioMC.analytic_ray_tracing").setLevel(logging.ERROR)

memory = Memory(location="__flight_utilities_cache", verbose=0)

def tqdm(*args, **kwargs):
    kwargs.setdefault("mininterval", 5)
    return tqdm_(*args, **kwargs)

# =============================================================================
# TRAVEL-TIME MAPS
# =============================================================================

def _load_ice_model(calibration_file):
    logger.warning("Loading ice model from %s", calibration_file)
    with lzma.open(calibration_file, "rt") as f:
        calibrated_ice_data = json.load(f)["additional_data"]["ice_model"]
    medium_args = calibrated_ice_data["args"]
    return medium_base.IceModelContinuousExpLayers(**medium_args)

def _compute_pair_tt_map(pos_a, pos_b, delay_a, delay_b, zeniths, azimuths, rt):
    tt_map = np.empty((zeniths.size, azimuths.size), dtype=np.float32)

    rt.set_start_and_end_point_no_swap(pos_a, pos_b)

    for iz, zen in enumerate(zeniths):
        for ia, az in enumerate(azimuths):
            tt_map[iz, ia] = rt.get_time_difference_plane_wave(zen, az) + delay_a - delay_b

    return tt_map

def _build_travel_time_map(det, ice_model, station_id, channels,
                           zeniths = np.linspace(1e-5, np.pi / 2, 90 * 6),
                           azimuths = np.linspace(0.0, 2 * np.pi, 360 * 6),
                           use_multiprocessing=False):

    from NuRadioMC.SignalProp import propagation
    art = propagation.get_propagation_module("analytic")
    rt = art(ice_model, compile_numba=True)

    tt_maps = {"zeniths": zeniths, "azimuths": azimuths}

    channel_pairs = list(itertools.combinations(channels, 2))
    channel_positions = {ch: det.get_relative_position(station_id, ch) \
                          for ch in channels}
    channel_delays = {ch: det.get_time_delay(station_id, ch) \
            for ch in channels}
    pair_args = (
        (channel_positions[ch_a], channel_positions[ch_b],
         channel_delays[ch_a], channel_delays[ch_b], zeniths, azimuths, rt)
        for ch_a, ch_b in channel_pairs
    )

    if use_multiprocessing:
        from joblib import Parallel, delayed

        maps = tqdm(Parallel(n_jobs=16, backend="loky")(
            delayed(_compute_pair_tt_map)(*args) for args in pair_args
        ),     total=len(channel_pairs),
)
    else:
        maps = (_compute_pair_tt_map(*args) for args in tqdm(pair_args, total=len(channel_pairs)))

    tt_maps.update(zip(channel_pairs, maps))
    return tt_maps



# =============================================================================
# XCORR RECONSTRUCTION
# =============================================================================

def deep_plane_reco(trace_by_channel, SNR_dict, fs, tt_maps, lags = None, normfact = None):
    
    corrs = []
    SNR_sum = np.sum(list(SNR_dict.values()))
    
    for ((ch_a, trace_a), (ch_b, trace_b)) in itertools.combinations(trace_by_channel.items(), 2):
        weight = 1.0
        #weight *= SNR_dict[ch_a] / SNR_sum
        #weight *= SNR_dict[ch_b] / SNR_sum
        
        if ch_a in [0,1,2,3]:
            weight*=0.25
        if ch_b in [0,1,2,3]:
            weight*=0.25
        
        if lags is None:
            n_samples = len(trace_a)
            lags = scipy.signal.correlation_lags(n_samples, n_samples, mode='full')

        if normfact is None:
            normfact = scipy.signal.correlate(np.ones_like(trace_a), np.ones_like(trace_b), mode = "full")

        delta_t_map = tt_maps[(ch_a, ch_b)]
        correlation = scipy.signal.correlate(
                trace_a, trace_b,
                mode="full"
                ) / normfact * weight

        expected_lags = delta_t_map * fs
        corrs.append(np.interp(expected_lags, lags, correlation, left=0.0, right=0.0))

    corr_map = np.mean(corrs, axis = 0)
    return corr_map

def _refine_peak_parabola(corr_map, corr_index, window_size=5):
    zenith_index, azimuth_index = corr_index
    half_window = window_size // 2
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer of at least 3")
    if not (
        half_window <= zenith_index < corr_map.shape[0] - half_window
        and half_window <= azimuth_index < corr_map.shape[1] - half_window
    ):
        return corr_index

    offsets = np.arange(-half_window, half_window + 1)
    zenith_offsets, azimuth_offsets = np.meshgrid(offsets, offsets, indexing="ij")
    samples = corr_map[
        zenith_index - half_window:zenith_index + half_window + 1,
        azimuth_index - half_window:azimuth_index + half_window + 1,
    ]
    design = np.column_stack([
        np.ones(samples.size), zenith_offsets.ravel(), azimuth_offsets.ravel(),
        zenith_offsets.ravel() ** 2, zenith_offsets.ravel() * azimuth_offsets.ravel(),
        azimuth_offsets.ravel() ** 2,
    ])
    _, zenith_slope, azimuth_slope, zenith_square, cross_term, azimuth_square = np.linalg.lstsq(
        design, samples.ravel(), rcond=None
    )[0]
    hessian = np.array([[2 * zenith_square, cross_term], [cross_term, 2 * azimuth_square]])
    if np.any(np.linalg.eigvalsh(hessian) >= 0):
        return corr_index

    offset = np.linalg.solve(hessian, [-zenith_slope, -azimuth_slope])
    if np.any(np.abs(offset) > half_window):
        return corr_index
    return np.asarray(corr_index, dtype=float) + offset

def _get_run_path(station_, run_id, inbox, new_dir):
    if inbox:
        return os.path.join(INBOX_RNOG_DATA, f"station{station_}/run{run_id}/combined.root")
    if new_dir:
        return os.path.join(NEW_DATA, f"station{station_}/run{run_id}")
    return os.path.join(RNOG_DATA, f"station{station_}/run{run_id}")

def _configure_reader(reader, run_path, force, lt):
    reader_kwargs = {
        "mattak_kwargs": {"backend": "uproot"},
        "apply_baseline_correction": None,
    }
    if force:
        reader.begin([run_path], select_triggers=["FORCE"], **reader_kwargs)
    elif lt:
        reader.begin([run_path], select_triggers=["LT"], **reader_kwargs)
    else:
        reader.begin([run_path], **reader_kwargs)

def _get_avg_SNR(station, channels = [0, 1, 2, 3]):
    SNRs = np.mean([station.get_channel(ch)[chp.SNR]["peak_2_peak_amplitude"] for ch in channels])
    return np.mean(SNRs)

def _get_max_SNR(station, channels = [0, 1, 2, 3]):
    SNRs = [station.get_channel(ch)[chp.SNR]["peak_2_peak_amplitude"] for ch in channels]
    return np.max(SNRs)

def _get_n_channels_SNR(station, channels = [0, 1, 2, 3], threshold=5.0):
    SNRs = [station.get_channel(ch)[chp.SNR]["peak_2_peak_amplitude"] for ch in channels]
    return sum(snr > threshold for snr in SNRs)

def get_SNRs(station, channels = [0, 1, 2, 3]):
    SNRs = [station.get_channel(ch)[chp.SNR]["peak_2_peak_amplitude"] for ch in channels]
    return SNRs

def _get_coherent_snr(station, channels):
    traces = [station.get_channel(ch).get_trace() for ch in channels]
    SNRs = [station.get_channel(ch)[chp.SNR]["peak_2_peak_amplitude"] for ch in channels]
    argmax = np.argmax(SNRs)
    trace_set = [traces[i] for i in range(len(traces)) if i != argmax]
    sum_trace = trace_utilities.get_coherent_sum(trace_set=trace_set, ref_trace=traces[argmax])
    rms = trace_utilities.get_split_trace_noise_RMS(sum_trace, segments=4, lowest=2)
    snr = trace_utilities.get_signal_to_noise_ratio(sum_trace, rms, 
        window_size=round(
            10*units.ns * station.get_channel(channels[argmax]
            ).get_sampling_rate()))
    return snr

def _run_deep_reco(channels, SNRs, station, det, tt_maps, run_id, event_id, ttype, config_dict=None, parabola_refine=True):

    # Trace preprocessing
    resample_factor = 8
    fs = station.get_channel(channels[0]).get_sampling_rate() * resample_factor
    
    
    def trace_preprocessor(trace):
        
        mod_trace = resample(trace, sampling_factor=resample_factor)

        pad_width = 512
        padded_trace = np.pad(
            mod_trace,
            (pad_width, pad_width),
            mode="constant",
            constant_values=0,
        )

        filtered_trace = butterworth_filter_trace(
            padded_trace,
            fs,
            [0.05, 0.5],
            order=8,
        )

        # Remove the padding after filtering.
        mod_trace = filtered_trace[pad_width:-pad_width]

        std = np.std(mod_trace)
        if std > 0:
            mod_trace = mod_trace / std

        return mod_trace

    trace_by_channel = {ch: trace_preprocessor(station.get_channel(ch).get_trace()) for ch in channels}

    
    SNR_dict = {ch: SNRs[i] for i, ch in enumerate(channels)}
    # Correlation map calculation
    corr_map = deep_plane_reco(trace_by_channel, SNR_dict, fs, tt_maps) 
   
    # Extraction of maximum-correlation direction
    corr_index = np.unravel_index(np.argmax(corr_map), corr_map.shape)
    
    # TODO: in principle could do a parabolic fit near the maximum
    best_result = (tt_maps["zeniths"][corr_index[0]], 
                   tt_maps["azimuths"][corr_index[1]], 
                   corr_map[corr_index])

    logger.info(
        "Event %s type %s run %s: deep corr z=%.2f a=%.2f corr=%.2f",
        event_id, ttype, run_id,
        np.degrees(best_result[0]), np.degrees(best_result[1]), best_result[2],
    )
    return best_result, corr_map


def process_run_deep(
    output_path,
    run_id,
    station_,
    channels=None,
    det_file=None,
    inbox=False,
    new_dir=False,
    FORCE=False,
    LT=False,
    config_dict=None,
    skymap_outdir="./skymaps",
    selected_events=None,
    tt_map_file=None,
):

    if channels is None:
        raise RuntimeError("Channels must be specified")

    run_path = _get_run_path(station_, run_id, inbox, new_dir)
    if not os.path.exists(run_path) or (
        os.path.isdir(run_path) and not os.listdir(run_path)
    ):
        logger.critical("Run path is missing or empty: %s", run_path)
        return None

    from NuRadioReco.detector.RNO_G.rnog_detector import Detector
    if det_file is not None:
        # Load calibrated RNO-G detector
        detector = Detector(detector_file=det_file, select_stations=station_)
        
        if station_ == 14:
            detector.update(pd.to_datetime("2025-05-01T00:00:00"))
        elif station_ == 22:
            detector.update(pd.to_datetime("2023-06-01T00:00:00"))
        else:
            detector.update(pd.to_datetime("2024-06-01T00:00:00"))


        # load calibrated ice model
        ice_model = _load_ice_model(det_file)
    else:
        detector = Detector(select_stations=station_)
        if station_ == 14:
            detector.update(pd.to_datetime("2025-05-01T00:00:00"))
        elif station_ == 22:
            detector.update(pd.to_datetime("2023-06-01T00:00:00"))
        else:
            detector.update(pd.to_datetime("2024-06-01T00:00:00"))

        # load standard ice model
        ice_model = medium.greenland_3exp_layered()

    if tt_map_file is not None and os.path.exists(tt_map_file):
        logger.info("Loading travel-time map from %s", tt_map_file)
        prep_map = load_map(tt_map_file)
    else:
        print("Travel-time map file %s not found. Building new travel-time map.", tt_map_file)
        logger.info("Building travel-time map for station %s", station_)
        prep_map = _build_travel_time_map(detector, ice_model, station_, channels)


    # set up all the standard NuRadio modules
    provider = dataProviderRNOG()
    reader = provider.reader
    reader.logger.setLevel(60)
    logging.getLogger("NuRadioMC").setLevel(logging.ERROR)
    logging.getLogger("NuRadioReco").setLevel(logging.CRITICAL)

    signal_reconstructor = channelSignalReconstructor()

    if config_dict is not None:
        logger.warning(f"Using custom config_dict: {config_dict}")

    # initialize all the NuRadio modules
    try:
        _configure_reader(reader, run_path, FORCE, LT)
        provider.channelBlockOffsetFitter.begin()
        provider.channelGlitchDetector.begin()
        provider.channelCableDelayAdder.begin()
        signal_reconstructor.begin()
    except Exception as error:
        logger.error("Reader initialization failed for %s: %s", run_path, error)
        return None

    from root_result import RootRunResult
    result = RootRunResult(output_path, run_id=run_id, station=station_)

    # begin loop over events
    for event in reader.run():

        event_id = event.get_id()
        if selected_events is not None and event_id not in selected_events:
            continue
        print(f"Now on event {event_id}")

        station = event.get_station()
        trigger_type = station.get_first_trigger().get_name()

        if not trigger_type:
            logger.critical("Event %s has an empty trigger type", event_id)
            sys.exit(1)

        station_time = station.get_station_time()
        if pd.isnull(station_time):
            logger.critical("Event %s has an invalid station time", event_id)
            sys.exit(1)

        signal_reconstructor.run(event, station, detector)
        
        SNRs = get_SNRs(station, channels=channels)
        
        max_SNR = max(SNRs)
        
        #if max_SNR < 5.0:
        #    continue
        avg_PA_SNR = np.mean(SNRs[0:4])
        
        if avg_PA_SNR < 3.0:
            continue
        n_channels_SNR = sum(snr > 4.0 for snr in SNRs)
        
        #if n_channels_SNR < 3:
        #    continue


        (zenith, azimuth, correlation), corr_map = _run_deep_reco(
            channels, SNRs, station, detector, prep_map,
            run_id, event_id, trigger_type, config_dict=config_dict,
        )
        coherent_snr = _get_coherent_snr(station, channels)
        
        if skymap_outdir is not None:  # Only plot for 20% of events to save time
            if not os.path.exists(skymap_outdir):
                os.makedirs(skymap_outdir)

            skymap_path = os.path.join(skymap_outdir, 
                                       f"station_{station_}_run_{run_id}_evt_{event_id}_corr_{correlation:.2f}.pdf")
            plot_skymap(skymap_path, corr_map, prep_map["zeniths"], prep_map["azimuths"])

        result.add_event(
                event_id,
                station_time.datetime64,
                trigger_types=trigger_type,
                PA_snr=avg_PA_SNR,
                snr=max_SNR,
                zeniths_deep_corr=zenith,
                azimuths_deep_corr=azimuth,
                corr_deep=correlation,
                n_channels_snr=n_channels_SNR,
                coherent_snr=coherent_snr,
        )

    signal_reconstructor.end()
    provider.end()

    result.close()

    return result

