import itertools
import json
import logging
import lzma
import os

import numpy as np
import scipy.signal

from NuRadioMC.utilities import medium_base
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import trace_utilities, units
from NuRadioReco.utilities.signal_processing import butterworth_filter_trace, resample

logger = logging.getLogger(__name__)


def tqdm(*args, **kwargs):
    kwargs.setdefault("mininterval", 5)
    #check if tqdm is available
    try:
        from tqdm.auto import tqdm as tqdm_
    except ImportError:
        tqdm_ = lambda x, **kwargs: x
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
                           zeniths = np.linspace(1e-5, np.pi / 2, 90 * 1),
                           azimuths = np.linspace(0.0, 2 * np.pi, 360 * 1),
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

        maps = tqdm(Parallel(n_jobs=8, backend="loky")(
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

def deep_plane_reco(trace_by_channel, fs, tt_maps, lags = None, normfact = None):
    
    corrs = []
    
    for ((ch_a, trace_a), (ch_b, trace_b)) in itertools.combinations(trace_by_channel.items(), 2):
        weight = 1.0
    
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



def _get_run_path(station_, run_id, inbox, new_dir):
    
    INBOX_RNOG_DATA = "/pnfs/ifh.de/acs/radio/diskonly/data/inbox/"
    RNOG_DATA = "/pnfs/ifh.de/acs/radio/diskonly/data/full/root/"
    NEW_DATA = "/pnfs/ifh.de/acs/radio/diskonly/new_data/"
    
    if inbox:
        return os.path.join(INBOX_RNOG_DATA, f"station{station_}/run{run_id}/combined.root")
    if new_dir:
        return os.path.join(NEW_DATA, f"station{station_}/run{run_id}")
    return os.path.join(RNOG_DATA, f"station{station_}/run{run_id}")


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

def _run_deep_reco(channels, station, tt_maps):

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

    

    # Correlation map calculation
    corr_map = deep_plane_reco(trace_by_channel, fs, tt_maps) 
   
    # Extraction of maximum-correlation direction
    corr_index = np.unravel_index(np.argmax(corr_map), corr_map.shape)
    
    # TODO: in principle could do a parabolic fit near the maximum
    best_result = (tt_maps["zeniths"][corr_index[0]], 
                   tt_maps["azimuths"][corr_index[1]], 
                   corr_map[corr_index])

    return best_result, corr_map



class PlaneWaveReconstructor():
    def __init__(self, station, detector, channels, ice_model, tt_map_path=None):
        """Initialize class
        """
        self.station = station
        self.detector = detector
        self.channels = channels
        self.ice_model = ice_model
        self.tt_map_path = tt_map_path

    def begin(self):
        """Begin the plane wave reconstructor"""
        if self.tt_map_path is not None:
            from travel_time_maps import load_map
            self.tt_map = load_map(self.tt_map_path)
        else:
            from travel_time_maps import map_from_dict
            self.tt_map = map_from_dict( _build_travel_time_map(self.detector, self.ice_model, self.station, self.channels) )

    
    @register_run()
    def run(self, evt, station, return_corr_map=False):
        """Run the plane wave reconstructor"""
        
        (zenith, azimuth, correlation), corr_map = _run_deep_reco(
            self.channels, station, self.tt_map,
        )
        if return_corr_map:
            return (zenith, azimuth, correlation), corr_map
        return (zenith, azimuth, correlation)


    def end(self):
        """Unused"""


def plot_skymap(outpath, corrmap, zen, az):
    import matplotlib.pyplot as plt
    import scipy.ndimage
    from matplotlib.gridspec import GridSpec


    fig = plt.figure(figsize = (5, 4), layout = "constrained")
    gs = GridSpec(1, 1, figure = fig)
    ax = fig.add_subplot(gs[0], projection="polar")

    cmax = np.max(np.abs(corrmap))

    im = ax.pcolormesh(az, zen, corrmap,
                       cmap='bwr', rasterized=True, vmin=-cmax, vmax=cmax)
    
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.05, pad = 0.03)
        # Run peak finder

    max_zenith_index, max_azimuth_index = np.unravel_index(
        np.argmax(corrmap), corrmap.shape
    )

    ax.scatter(
        az[max_azimuth_index],
        zen[max_zenith_index],
        edgecolor="green",
        facecolor="none",
        s=100,
        label="Max Correlation",
        zorder=10,
    )
    ax.annotate(
        f"{corrmap[max_zenith_index, max_azimuth_index]:.2f}",
        (az[max_azimuth_index], zen[max_zenith_index]),
        textcoords="offset points",
        xytext=(0, 5),
        ha='center',
        fontsize=8,
        color="green",
        zorder=11,
    )
    
    data_max = scipy.ndimage.maximum_filter(corrmap, size=10)
    mask = (corrmap == data_max) & (corrmap > 0.9 * cmax)
    y, x = mask.nonzero()
    # remove maxima from secondary peaks
    max_mask = (y == max_zenith_index) & (x == max_azimuth_index)
    x = x[~max_mask]
    y = y[~max_mask]
    
    if len(x) != len(y):
        print(f"Warning: Found {len(x)} secondary peaks but {len(y)} zenith indices. This should not happen.")
        return 
    
    value = corrmap[y, x]
    
    
    print(f"Found {len(x)} peaks at 2-sigma")
    
    ax.scatter(az[x], zen[y], edgecolor="black", facecolor="none", s=50, label="Secondary Peaks $2\\sigma$", zorder=10)
    for i in range(len(x)):
        ax.annotate(f"{value[i]:.2f}", (az[x[i]], zen[y[i]]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8, color="black", zorder=11)
    
    fig.legend(loc="outside upper right", fontsize=10)
    

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(
        ["E (0°)", "NE (45°)", "N (90°)", "NW (135°)",
         "W (180°)", "SW (225°)", "S (270°)", "SE (315°)"],
        fontsize=10, 
    )
    ax.set_rlim(0, np.pi / 2)
    rticks = [np.deg2rad(r) for r in (0, 15.1, 30.1, 45.1, 60.1, 75.1, 90)]
    ax.set_rticks(rticks)
    
    ax.set_yticklabels([f"{int(np.degrees(r))}°" for r in rticks],
                       color="black", fontsize=10, zorder=10)

    cbar.set_label("Correlation")

    fig.savefig(outpath)
    
    plt.close()
