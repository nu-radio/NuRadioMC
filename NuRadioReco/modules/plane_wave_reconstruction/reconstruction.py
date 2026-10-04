import itertools
import json
import logging
import lzma
import os

import numpy as np
import scipy.fft
import time

from NuRadioMC.utilities import medium_base
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import trace_utilities, units
from NuRadioReco.utilities.signal_processing import butterworth_filter_trace, resample

logger = logging.getLogger(__name__)

try:
    import numba

    @numba.njit(parallel=True)
    def _add_interp(out, tt_map, correlation, fs, lag0):
        """out += correlation linearly interpolated at lag = tt_map * fs (lag0 = index of zero lag); 0 outside."""
        n_lags = correlation.size
        one = np.float32(1.0)
        for iz in numba.prange(tt_map.shape[0]):
            for ia in range(tt_map.shape[1]):
                x = tt_map[iz, ia] * fs + lag0
                i0 = int(np.floor(x))
                if 0 <= i0 < n_lags - 1:
                    f = x - np.float32(i0)
                    out[iz, ia] += correlation[i0] * (one - f) + correlation[i0 + 1] * f
except ImportError:
    numba = None


def tqdm(*args, **kwargs):
    kwargs.setdefault("mininterval", 5)
    #check if tqdm is available, if not, use a dummy tqdm
    try:
        from tqdm.auto import tqdm as tqdm_
    except ImportError:
        tqdm_ = lambda x, **kwargs: x
    return tqdm_(*args, **kwargs)

# =============================================================================
# TRAVEL-TIME MAPS
# =============================================================================

def _load_ice_model(calibration_file):
    """
    load the ice model from the given calibration file.
    """
    logger.warning("Loading ice model from %s", calibration_file)
    with lzma.open(calibration_file, "rt") as f:
        calibrated_ice_data = json.load(f)["additional_data"]["ice_model"]
    medium_args = calibrated_ice_data["args"]
    return medium_base.IceModelContinuousExpLayers(**medium_args)

def _compute_pair_tt_map(pos_a: np.ndarray, pos_b: np.ndarray, delay_a: float, delay_b: float, zeniths: np.ndarray, azimuths: np.ndarray, rt) -> np.ndarray:
    """
    Compute the travel-time map for a pair of channels given their positions, delays, and the ray-tracing object.
    Parameters
    ----------
    pos_a : np.ndarray
        (x, y, z) position of channel A.
    pos_b : np.ndarray
        (x, y, z) position of channel B.
    delay_a : float
        Time delay of channel A.
    delay_b : float
        Time delay of channel B.
    zeniths : np.ndarray
        Array of zenith angles.
    azimuths : np.ndarray
        Array of azimuth angles.
    rt : RayTracing
        Ray-tracing object.
    
    Returns
    -------
    tt_map : np.ndarray
        Travel-time map for the channel pair.
    """
    tt_map = np.empty((zeniths.size, azimuths.size), dtype=np.float32)

    rt.set_start_and_end_point_no_swap(pos_a, pos_b)

    for iz, zen in enumerate(zeniths):
        for ia, az in enumerate(azimuths):
            tt_map[iz, ia] = rt.get_time_difference_plane_wave(zen, az) + delay_a - delay_b

    return tt_map

def _build_travel_time_map(det, ice_model, station_id: int, channels: list[int],
                           zeniths_steps: int = 90,
                           azimuths_steps: int = 360,
                           use_multiprocessing: bool = False):
    """
    Build the travel-time map for all pairs of channels in the given station.
    Parameters
    ----------
    det : Detector
        The detector object containing station and channel information.
    ice_model : IceModel
        The ice model used for ray tracing. 
        The ice model used for ray tracing.
    station_id : int
        The ID of the station for which to build the travel-time map.
    channels : list of int
        List of channel indices to include in the travel-time map.
    zeniths_steps : int, optional
        Number of steps for the zenith angle array. Default is 90.
    azimuths_steps : int, optional
        Number of steps for the azimuth angle array. Default is 360.
    use_multiprocessing : bool, optional
        Whether to use multiprocessing for computing the travel-time maps. Default is False.

    Returns
    -------
    tt_maps : dict
        Dictionary containing the travel-time maps for all pairs of channels, along with the zenith and azimuth arrays.
    """
    from NuRadioMC.SignalProp import propagation
    art = propagation.get_propagation_module("analytic")
    rt = art(ice_model, compile_numba=True)

    zeniths = np.linspace(1e-5, np.pi / 2, zeniths_steps)
    azimuths = np.linspace(0.0, 2 * np.pi, azimuths_steps)
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

def deep_plane_reco(trace_by_channel: dict[int, np.ndarray], fs: float, tt_maps: dict, use_numba: bool = False):
    """
    Perform deep plane wave reconstruction by computing the cross-correlation map for all pairs of channels.

    Parameters
    ----------
    trace_by_channel : dict
        Dictionary mapping channel indices to their corresponding traces.
    fs : float
        Sampling frequency of the traces.
    tt_maps : dict
        Travel-time maps for all pairs of channels.
    use_numba : bool, optional
        Use a numba kernel for the lag interpolation. Default is False.

    Returns
    -------
    corr_map : np.ndarray
        Cross-correlation map averaged over all pairs of channels.
    """
    
    n_samples = len(next(iter(trace_by_channel.values())))
    n_fft = scipy.fft.next_fast_len(2 * n_samples - 1, real=True)
    fs = np.float32(fs)

    # lag axis in samples: -(N-1) ... N-1
    lags = np.arange(-(n_samples - 1), n_samples, dtype=np.float32)
    # correct for the varying number of overlapping samples at each lag: 1 / (N - |lag|)
    inv_overlap = (1.0 / (n_samples - np.abs(lags))).astype(np.float32)

    # one forward FFT per channel and one inverse FFT per pair, both batched (complex64 / float32)
    channels = list(trace_by_channel)
    traces = np.stack([trace_by_channel[ch] for ch in channels]).astype(np.float32)
    spec = scipy.fft.rfft(traces, n=n_fft, axis=1, workers=-1)
    idx_a, idx_b = np.array(list(itertools.combinations(range(len(channels)), 2))).T
    r = scipy.fft.irfft(spec[idx_a] * np.conj(spec[idx_b]), n=n_fft, axis=1, workers=-1)
    # circular -> linear lag order: lags -(N-1)..-1 sit at the end, 0..N-1 at the start
    xcorrs = np.concatenate((r[:, n_fft - (n_samples - 1):], r[:, :n_samples]), axis=1) * inv_overlap

    corrs = np.zeros((tt_maps["zeniths"].size, tt_maps["azimuths"].size), dtype=np.float32)
    n_pairs = 0
    # loop over all unique pairs of channels
    for p, (i_a, i_b) in enumerate(zip(idx_a, idx_b)):
        ch_a, ch_b = channels[i_a], channels[i_b]
        weight = np.float32(1.0)
        
        # phased array channels get lower weight since they are so close together
        if ch_a in [0,1,2,3]:
            weight*=0.25
        if ch_b in [0,1,2,3]:
            weight*=0.25
        
        # lag axis in n_lag_samples: -(N-1) ... N-1
        # (computed once above)
            
        # (overlap normalisation applied above)
        
        # 2d array: expected dt(zenith, azimuth) for this pair, in time units
        delta_t_map = tt_maps[(ch_a, ch_b)]
        
        # 1d array: cross-correlation of the two traces, xcorr(n_lag_samples)
        correlation = xcorrs[p] * weight

        # expected lag in samples: dt * fs = n_lag_samples
        # we go from 2D dt map -> 2D n_lag_samples map
        expected_lags = delta_t_map * fs
        # interpolate, get the xcorr value at the expected lags for this pair
        # with 2D input np.interp evaluates elementwise and returns 
        # an array with same shape as first argument
        # we go from 2D n_lag_samples map -> 2D xcorr map
        
        if use_numba:
            _add_interp(corrs, delta_t_map, correlation, fs, np.float32(n_samples - 1))
        else:
            corrs += np.interp(expected_lags, lags, correlation, left=0.0, right=0.0)
        n_pairs += 1
    
    corr_map = corrs / n_pairs
    return corr_map


def _get_coherent_snr(station, channels):
    """
    Compute the coherent signal-to-noise ratio (SNR) for a given station and set of channels.

    Parameters
    ----------
    station : object
        The station object containing channel data.
    channels : list
        List of channel indices to consider for the coherent SNR calculation.

    Returns
    -------
    snr : float
        The coherent signal-to-noise ratio for the given channels.
    """
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

def _run_deep_reco(channels: list[int], station, tt_maps: dict, trace_preprocessor=None, use_numba=False):
    """
    Run the deep plane wave reconstruction for a given set of channels and station.

    Parameters
    ----------
    channels : list
        List of channel indices to consider for the reconstruction.
    station : object
        The station object containing channel data.
    tt_maps : dict
        Travel-time maps for all pairs of channels.

    Returns
    -------
    best_result : tuple
        The best reconstruction result as a tuple (zenith, azimuth, correlation).
    corr_map : np.ndarray
        The cross-correlation map used for the reconstruction.
    """
    #TODO: allow user to pss custom channel weights

    # Trace preprocessing
    resample_factor = 8
    fs = station.get_channel(channels[0]).get_sampling_rate() * resample_factor
    
    if trace_preprocessor is None:
        
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
    corr_map = deep_plane_reco(trace_by_channel, fs, tt_maps, use_numba=use_numba)
   
    # Extraction of maximum-correlation direction
    corr_index = np.unravel_index(np.argmax(corr_map), corr_map.shape)
    
    # TODO: in principle could do a parabolic fit near the maximum
    best_result = (tt_maps["zeniths"][corr_index[0]], 
                   tt_maps["azimuths"][corr_index[1]], 
                   corr_map[corr_index])

    return best_result, corr_map



class PlaneWaveReconstructor():
    """
    Plane wave reconstructor class.
    """
    def __init__(self, station, detector, channels, ice_model, tt_map_path=None, use_numba=True):
        """Initialize class

        use_numba : bool, optional
            Use the numba kernel for the lag interpolation (requires numba). Default is False.
        """
        if use_numba and numba is None:
            raise ImportError("use_numba=True requires numba to be installed")
        self.use_numba = use_numba
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
    def run(self, evt, station, return_corr_map=False, print_execution_time=True):
        """Run the plane wave reconstructor
        Parameters
        ----------
        evt : Event object
            The event to process.
        station : Station object
            The station containing the traces.
        return_corr_map : bool, optional
            Whether to return the correlation map, by default False.

        Returns
        -------
        tuple
            (zenith, azimuth, correlation) and optionally the correlation map.
        """
        t_start = time.perf_counter()

        (zenith, azimuth, correlation), corr_map = _run_deep_reco(
            self.channels, station, self.tt_map, use_numba=self.use_numba,
        )
        if print_execution_time:
            t_end = time.perf_counter()
            print(f"Execution time: {t_end - t_start:.6f} seconds")
        if return_corr_map:
            return (zenith, azimuth, correlation), corr_map
        return (zenith, azimuth, correlation)


    def end(self):
        """Unused"""


def plot_skymap(outpath, corrmap, zen, az):
    """
    Plot the sky map of the correlation map with the reconstructed zenith and azimuth angles.

    Parameters
    ----------
    outpath : str
        The path to save the plot.
    corrmap : 2D numpy array
        The correlation map.
    zen : 1D numpy array
        The zenith angles corresponding to the correlation map.
    az : 1D numpy array
        The azimuth angles corresponding to the correlation map.
    """
    
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
