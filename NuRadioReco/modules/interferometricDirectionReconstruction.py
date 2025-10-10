import matplotlib.pyplot as plt
import numpy as np
import itertools
from datetime import datetime
import os
import yaml
import logging
import h5py
import pickle
import hashlib

from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units
from NuRadioReco.detector.detector import Detector
from NuRadioReco.framework.parameters import stationParameters as stnp

from scipy.signal import correlate, correlation_lags
from scipy.interpolate import RegularGridInterpolator
logger = logging.getLogger("NuRadioReco.modules.interferometricDirectionReconstruction")

"""
This module provides a class for directional reconstruction by fitting time delays between channels to predifined time delay maps.
Usage requires pre-calculated time delay tables for each channel and configuration file specifying reconstruction parameters.
"""

def get_delay_matrix_cache_key(station, channels, limits, step_sizes, coord_system, fixed_coord, rec_type, cable_delays):
    """
    Generate a unique cache key for delay matrix configuration.
    Delay matrices only depend on geometry and configuration, not on the event data.
    """
    channels_sorted = tuple(sorted(channels))
    
    config_tuple = (
        station,
        channels_sorted,
        tuple(limits),
        tuple(step_sizes),
        coord_system,
        fixed_coord,
        rec_type,
        tuple(sorted((ch, round(delay, 6)) for ch, delay in cable_delays.items()))  # Round to avoid float precision issues for cacheing purposes only
    )
    
    config_str = str(config_tuple)
    cache_key = hashlib.md5(config_str.encode()).hexdigest()[:16]
    
    return cache_key

def get_delay_matrix_cache_path(cache_key, station):
    cache_dir = os.path.expanduser("~/.cache/nuradio_delay_matrices")
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except:
        cache_dir = f"/tmp/nuradio_delay_matrices_{os.getuid()}"
        os.makedirs(cache_dir, exist_ok=True)
    
    filename = f"station{station}_delays_{cache_key}.pkl"
    return os.path.join(cache_dir, filename)

def load_delay_matrices_from_cache(cache_key, station):
    filepath = get_delay_matrix_cache_path(cache_key, station)
    
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'rb') as f:
            cache_data = pickle.load(f)
        
        logger.info(f"Loaded delay matrices from cache: {filepath}")
        return cache_data['delay_matrices']
    except Exception as e:
        logger.warning(f"Failed to load cache: {e}")
        return None

def save_delay_matrices_to_cache(delay_matrices, cache_key, station, config_info=None):
    filepath = get_delay_matrix_cache_path(cache_key, station)
    
    cache_data = {
        'delay_matrices': delay_matrices,
        'cache_key': cache_key,
        'station': station,
        'timestamp': datetime.now().isoformat(),
        'config_info': config_info
    }
    
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved delay matrices to cache: {filepath}")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")


def load_rz_interpolator(table_filename):
    file = np.load(table_filename)
    travel_time_table = file['data']
    r_range = file['r_range_vals']
    z_range = file['z_range_vals']

    interpolator = RegularGridInterpolator(
        (r_range, z_range), travel_time_table, method="linear", bounds_error=False, fill_value=-np.inf
    )
    return interpolator

def get_t_delay_matrices(station, config, src_posn_enu_matrix, ant_locs, cable_delays):

    ch_pairs = list(itertools.combinations(config['channels'], 2))
    time_delay_matrices = []

    data_dir = config['time_delay_tables'] 
    outdir = data_dir + f"station{station}/"

    interpolators = {}
    for ch in set(itertools.chain(*ch_pairs)):
        #table_file = f"{outdir}ch{ch}_rz_table_R1_1600Z-1600_200.npz"
        table_file = f"{outdir}ch{ch}_rz_table_rel_ant.npz"
        interpolators[ch] = load_rz_interpolator(table_file)

    for ch1, ch2 in ch_pairs:
        pos1 = ant_locs[ch1]
        pos2 = ant_locs[ch2]
        interp1 = interpolators[ch1]
        interp2 = interpolators[ch2]

        # Compute coordinates relative to each antenna (both r and z)
        # r: horizontal distance from antenna
        # z: vertical offset from antenna (tables already use absolute z)
        r_rel_ch1 = np.linalg.norm(src_posn_enu_matrix[:, :, :2] - pos1[:2], axis=2)
        z_rel_ch1 = src_posn_enu_matrix[:, :, 2]
        coords_rel_ch1 = np.stack((r_rel_ch1, z_rel_ch1), axis=-1)
        
        r_rel_ch2 = np.linalg.norm(src_posn_enu_matrix[:, :, :2] - pos2[:2], axis=2)
        z_rel_ch2 = src_posn_enu_matrix[:, :, 2]
        coords_rel_ch2 = np.stack((r_rel_ch2, z_rel_ch2), axis=-1)

        travel_times_to_ch1 = interp1(coords_rel_ch1)
        travel_times_to_ch2 = interp2(coords_rel_ch2)
        
        cable_delay_diff = cable_delays[ch1] - cable_delays[ch2]
        time_delay_matrix = travel_times_to_ch1 - travel_times_to_ch2 + cable_delay_diff

        time_delay_matrices.append(time_delay_matrix)

    return time_delay_matrices

def correlator(times, v_array_pairs, delay_matrices, apply_hann_window=False, use_hilbert=False):

    volt_corrs = []
    time_lags_list = []
    
    channels = list(range(len(times)))
    channel_pairs = list(itertools.combinations(channels, 2))
    
    # Pre-compute overlap normalization only once per unique length pair
    overlap_norms = {}
    
    for pair_idx, (v1, v2) in enumerate(v_array_pairs):
        len1, len2 = len(v1), len(v2)
        len_key = (len1, len2)
        
        ch1_idx, ch2_idx = channel_pairs[pair_idx]
        t1, t2 = times[ch1_idx], times[ch2_idx]
        
        dt1 = t1[1] - t1[0] if len(t1) > 1 else 1.0
        dt2 = t2[1] - t2[0] if len(t2) > 1 else 1.0
        dt = min(dt1, dt2)
        
        if len_key not in overlap_norms:
            overlap_norms[len_key] = correlate(np.ones(len1), np.ones(len2), mode='full')
        
        if use_hilbert:
            from scipy.signal import hilbert
            v1_proc = np.abs(hilbert(v1))
            v2_proc = np.abs(hilbert(v2))
        else:
            v1_proc = v1
            v2_proc = v2
        
        if apply_hann_window:
            window1 = np.hanning(len(v1_proc))
            window2 = np.hanning(len(v2_proc))
            v1_proc = v1_proc * window1
            v2_proc = v2_proc * window2
        
        corr = correlate(v1_proc, v2_proc, mode='full', method='auto')
        corr_normalized = corr / overlap_norms[len_key]
        volt_corrs.append(corr_normalized)
        
        lags = correlation_lags(len(v1), len(v2), mode="full")
        time_lags = lags * dt
        time_lags_list.append(time_lags)

    pair_corr_matrices = []

    for pair_idx, (volt_corr, time_lags, time_delay) in enumerate(zip(volt_corrs, time_lags_list, delay_matrices)):
        valid_mask = ~np.isnan(time_delay)
        
        pair_corr_matrix = np.full_like(time_delay, np.nan)
        
        if np.any(valid_mask):
            valid_delays = time_delay[valid_mask].flatten()
            interp_corr = np.interp(valid_delays, time_lags, volt_corr)
            pair_corr_matrix[valid_mask] = interp_corr
        
        pair_corr_matrices.append(pair_corr_matrix)    

    mean_corr_matrix = np.nanmean(pair_corr_matrices, axis=0)
    max_corr = np.nanmax(mean_corr_matrix)

    return mean_corr_matrix, max_corr


class interferometricDirectionReconstruction():
    """
    This module performs directional reconstruction by fitting time delays between channels to pre-defined time delay maps.
    """
    def __init__(self):
        self._cable_delay_cache = {}
        self._delay_matrix_cache = {}
        self._positions_cache = {}
        self._station_delay_matrices = {}
        self.begin()

    def begin(self, preload_cache_for_station=None, config=None, det=None):
        """
        Initialize the module, optionally preloading delay matrices and positions from disk cache.
        
        Parameters
        ----------
        preload_cache_for_station : int, optional
            Station ID to preload delay matrices for. If provided along with config,
            will load existing cache files into memory to avoid repeated disk I/O.
        config : dict or str, optional
            Configuration to use for generating the cache key. Required if preload_cache_for_station is set.
        det : detector.Detector, optional
            Detector object for precomputing positions and delay matrices.
        """
        if preload_cache_for_station is not None and config is not None:
            if isinstance(config, str):
                with open(config, "r") as f:
                    config = yaml.safe_load(f)
            
            if det is not None:
                self._precompute_station_data(preload_cache_for_station, config, det)
            
            cache_dir = os.path.expanduser("~/.cache/nuradio_delay_matrices")
            if os.path.exists(cache_dir):
                import glob
                cache_files = glob.glob(f"{cache_dir}/station{preload_cache_for_station}_delays_*.pkl")
                
                for cache_file in cache_files:
                    try:
                        with open(cache_file, 'rb') as f:
                            cache_data = pickle.load(f)
                        cache_key = cache_data['cache_key']
                        delay_matrices = cache_data['delay_matrices']
                        self._delay_matrix_cache[cache_key] = delay_matrices
                    except Exception as e:
                        logger.warning(f"Failed to preload {os.path.basename(cache_file)}: {e}")

    def _precompute_station_data(self, station_id, config, det):
        """
        Precompute positions and delay matrices for a station/config combination.
        
        Parameters
        ----------
        station_id : int
            Station ID
        config : dict
            Configuration dictionary
        det : detector.Detector
            Detector object
        """
        coord_system = config['coord_system']
        rec_type = config['rec_type']
        fixed_coord = config['fixed_coord']
        channels = config['channels']
        limits = config['limits']
        step_sizes = config['step_sizes']

        station_info = StationInfo(station_id, det)
        positions = Positions(
            station_info,
            limits,
            step_sizes,
            coord_system,
            fixed_coord,
            rec_type,
        )

        if config.get('apply_cable_delays', True):
            if station_id not in self._cable_delay_cache:
                self._cable_delay_cache[station_id] = {
                    ch: det.get_cable_delay(station_id, ch) / units.ns
                    for ch in range(24)
                }
            cable_delays = {ch: self._cable_delay_cache[station_id][ch] for ch in channels}
        else:
            cable_delays = {ch: 0.0 for ch in channels}

        cache_key = get_delay_matrix_cache_key(
            station_id,
            channels,
            limits,
            step_sizes,
            coord_system,
            fixed_coord,
            rec_type,
            cable_delays
        )
        
        if cache_key not in self._delay_matrix_cache:
            delay_matrices = load_delay_matrices_from_cache(cache_key, station_id)
            
            if delay_matrices is None:
                src_posn_enu_matrix, _, grid_tuple = positions.get_source_enu_matrix()
                delay_matrices = get_t_delay_matrices(
                    station_info.station, config, src_posn_enu_matrix, positions.ant_locs, cable_delays
                )
                
                config_info = {
                    'station': station_id,
                    'channels': channels,
                    'limits': limits,
                    'coord_system': coord_system,
                    'rec_type': rec_type,
                    'fixed_coord': fixed_coord
                }
                save_delay_matrices_to_cache(delay_matrices, cache_key, station_id, config_info)
            
            self._delay_matrix_cache[cache_key] = delay_matrices

        # Cache the precomputed data for this station/config
        station_config_key = (station_id, str(config))
        self._station_delay_matrices[station_config_key] = {
            'positions': positions,
            'delay_matrices': self._delay_matrix_cache[cache_key],
            'cache_key': cache_key
        }

    @register_run()
    def run(self, evt, station, det, config, corr_map=False):
        """
        args:
            evt: NuRadioReco.framework.event.Event
                The event to process.
            station: NuRadioReco.framework.station.Station
                The station to process.
            det: NuRadioReco.detector.detector.Detector
                The detector object containing the detector geometry and properties.
            config: dict or str
                Configuration dictionary or path to YAML configuration file specifying reconstruction parameters.
            corr_map: bool
                If True, generates and saves a correlation map plot for the event.
        """
        
        if isinstance(config, str):
            config = self.load_config(config)

        station_id = station.get_id()
        station_config_key = (station_id, str(config))
        
        positions = self._station_delay_matrices[station_config_key]['positions']
        delay_matrices = self._station_delay_matrices[station_config_key]['delay_matrices']

        channels = config['channels']
        volt_arrays = []
        time_arrays = []
        
        for ch in channels:
            channel = station.get_channel(ch)
            trace = channel.get_trace()
            
            if config.get('apply_waveform_scaling', True):
                if np.max(trace) != 0:
                    trace = trace / np.max(trace)
                if np.std(trace) != 0:
                    trace = trace / np.std(trace)
                trace = trace - np.mean(trace)
            
            volt_arrays.append(trace)
            time_arrays.append(channel.get_times())

        v_array_pairs = list(itertools.combinations(volt_arrays, 2))

        corr_matrix, max_corr = correlator(
            time_arrays, v_array_pairs, delay_matrices,
            apply_hann_window=config.get('apply_hann_window', False),
            use_hilbert=config.get('use_hilbert_envelope', False)
        )
        
        rec_coord0, rec_coord1 = positions.get_rec_locs_from_corr_map(
            corr_matrix
        )

        coord0_alt, coord1_alt, alt_indices, exclusion_bounds = None, None, None, None
        if config.get('find_alternate_reco', False):
            exclude_radius = config.get('alternate_exclude_radius_deg', 5.0)
            result = find_alternate_coordinate(corr_matrix, positions, exclude_radius)
            if result[0] is not None and result[1] is not None:
                coord0_alt, coord1_alt, alt_indices, exclusion_bounds = result
        
        if corr_map == True:
            plot_kwargs = {}
            if coord0_alt is not None and coord1_alt is not None:
                plot_kwargs['coord0_alt'] = coord0_alt
                plot_kwargs['coord1_alt'] = coord1_alt
                plot_kwargs['alt_indices'] = alt_indices
            if exclusion_bounds is not None:
                plot_kwargs['exclusion_bounds'] = exclusion_bounds
            self.plot_corr_map(corr_matrix, positions, evt=evt, config=config, **plot_kwargs)

        coord_system = config['coord_system']
        step_sizes = config['step_sizes']
        
        if coord_system == "cylindrical":
            num_rows_to_10m = int(np.ceil(10 / abs(step_sizes[1])))
            surface_corr= self.get_surf_corr(corr_matrix, num_rows_to_10m)
            station.set_parameter(stnp.rec_surf_corr, surface_corr)
            
        # elif coord_system == "spherical":
        #     num_rows_to_10m = 10
        #     surface_corr= self.get_surf_corr(corr_matrix, num_rows_to_10m)
        # else:
        #     surface_corr = -1.0

        station.set_parameter(stnp.rec_max_correlation, max_corr)

        station.set_parameter(stnp.rec_coord0, rec_coord0)
        station.set_parameter(stnp.rec_coord1, rec_coord1)

        if coord0_alt is not None and coord1_alt is not None:
            station.set_parameter(stnp.rec_coord0_alt, coord0_alt)
            station.set_parameter(stnp.rec_coord1_alt, coord1_alt)
        else:
            station.set_parameter(stnp.rec_coord0_alt, np.nan)
            station.set_parameter(stnp.rec_coord1_alt, np.nan)

        rec_type = config['rec_type']
        
        if coord_system == "cylindrical":
            if rec_type == "phiz":
                # For phiz: coord0 = φ (azimuth), coord1 = z (depth)
                station.set_parameter(stnp.rec_azimuth, rec_coord0)
                station.set_parameter(stnp.rec_z, rec_coord1)
            elif rec_type == "rhoz":
                # For rhoz: coord0 = ρ (radius), coord1 = z (depth)
                station.set_parameter(stnp.rec_rho, rec_coord0)
                station.set_parameter(stnp.rec_z, rec_coord1)
        elif coord_system == "spherical":
            # For spherical: coord0 = φ (azimuth), coord1 = θ (zenith)
            station.set_parameter(stnp.rec_azimuth, rec_coord0)
            station.set_parameter(stnp.rec_zenith, rec_coord1)
    
    def end(self):
        pass

    def load_config(self, config_file):
        with open(config_file, "r") as f:
            return yaml.safe_load(f)

    def get_surf_corr(self, corr_map, num_rows_for_10m):
        surf_corr = np.nanmax(corr_map[:num_rows_for_10m])

        return surf_corr

    def plot_corr_map(
        self,
        corr_matrix, positions, 
        file_name=None,
        evt=None, 
        config=None,
        show_actual_pulser=False,
        show_rec_pulser=True,
        **kwargs):

        run_number = evt.get_run_number()
        event_number = evt.get_id()
        station = evt.get_station()
        station_id = station.get_id()

        mycmap = plt.get_cmap("RdBu_r")
        mycmap.set_bad(color='black')

        plt.figure(figsize=(12, 8))
        fig, ax = plt.subplots()

        x = np.linspace(config['limits'][0], config['limits'][1], corr_matrix.shape[1] + 1)
        y = np.linspace(config['limits'][2], config['limits'][3], corr_matrix.shape[0] + 1)
        
        x_edges = np.linspace(
            x[0] - (x[1] - x[0]) / 2,
            x[-1] + (x[1] - x[0]) / 2,
            corr_matrix.shape[1] + 1,
        )
        y_edges = np.linspace(
            y[0] - (y[1] - y[0]) / 2,
            y[-1] + (y[1] - y[0]) / 2,
            corr_matrix.shape[0] + 1,
        )
        
        c = ax.pcolormesh(
            x_edges,
            y_edges,
            corr_matrix,
            cmap=mycmap,
            vmin=np.nanmin(corr_matrix),
            vmax=np.nanmax(corr_matrix),
            rasterized=True,
        )

        x_midpoints = (x_edges[:-1] + x_edges[1:]) / 2
        y_midpoints = (y_edges[:-1] + y_edges[1:]) / 2

        max_corr_value = np.nanmax(corr_matrix)
        max_corr_indices = np.unravel_index(
            np.nanargmax(corr_matrix), corr_matrix.shape
        )
        max_corr_x = x_midpoints[max_corr_indices[1]]
        max_corr_y = y_midpoints[max_corr_indices[0]]
        
        if positions.coord_system == "cylindrical":
            if positions.rec_type == "phiz":
                legend_label = f"Max corr: {max_corr_value:.2f} at ({int(max_corr_x)}°, {int(max_corr_y)}m)"
            elif positions.rec_type == "rhoz":
                legend_label = f"Max corr: {max_corr_value:.2f} at ({int(max_corr_x)}m, {int(max_corr_y)}m)"
        elif positions.coord_system == "spherical":
            legend_label = f"Max corr: {max_corr_value:.2f} at ({int(max_corr_x)}°, {int(config['limits'][3] - max_corr_y)}°)"
        
        ax.plot(
            max_corr_x,
            max_corr_y,
            "o",
            markersize=10,
            color="lime",
            label=legend_label,
        )

        if 'coord0_alt' in kwargs and 'coord1_alt' in kwargs:
            coord0_alt = kwargs['coord0_alt']
            coord1_alt = kwargs['coord1_alt']
            
            if coord0_alt is not None and coord1_alt is not None and not np.isnan(coord0_alt) and not np.isnan(coord1_alt):
                if 'alt_indices' in kwargs and kwargs['alt_indices'] is not None:
                    alt_idx0, alt_idx1 = kwargs['alt_indices']

                    alt_max_x = x_midpoints[alt_idx0]
                    alt_max_y = y_midpoints[alt_idx1]
                    alt_corr_val = corr_matrix[alt_idx1, alt_idx0]

                    if positions.rec_type == "phiz":
                        alt_corr_label = f"Alt max: {alt_corr_val:.2f} at ({alt_max_x:.0f}°, {alt_max_y:.1f}m)"
                    else:
                        alt_corr_label = f"Alt max: {alt_corr_val:.2f} at ({alt_max_x:.1f}m, {alt_max_y:.1f}m)"
                    
                    ax.plot(
                        alt_max_x,
                        alt_max_y,
                        "o",
                        markersize=5,
                        color="lime",
                        fillstyle="none",
                        markeredgewidth=1,
                        label=alt_corr_label,
                    )
                else:
                    if positions.coord_system == "cylindrical" and positions.rec_type == "phiz":
                        coord0_alt_val = coord0_alt / units.deg
                        coord1_alt_val = coord1_alt / units.m
                    elif positions.coord_system == "cylindrical" and positions.rec_type == "rhoz":
                        coord0_alt_val = coord0_alt / units.m
                        coord1_alt_val = coord1_alt / units.m
                    else:
                        coord0_alt_val = coord0_alt / units.deg
                        coord1_alt_val = coord1_alt / units.deg
                    
                    if 'max_corr_alt' in kwargs and kwargs['max_corr_alt'] is not None:
                        alt_corr_label = f"Alt max: {kwargs['max_corr_alt']:.2f}"
                    else:
                        try:
                            alt_x_idx = np.argmin(np.abs(x_midpoints - coord0_alt_val))
                            alt_y_idx = np.argmin(np.abs(y_midpoints - coord1_alt_val))
                            if 0 <= alt_x_idx < corr_matrix.shape[1] and 0 <= alt_y_idx < corr_matrix.shape[0]:
                                alt_corr_val = corr_matrix[alt_y_idx, alt_x_idx]
                                alt_corr_label = f"Alt max: {alt_corr_val:.2f}"
                            else:
                                alt_corr_label = "Alt max"
                        except:
                            alt_corr_label = "Alt max"
                    
                    ax.plot(
                        coord0_alt_val,
                        coord1_alt_val,
                        "o",
                        markersize=5,
                        color="lime",
                        fillstyle="none",
                        markeredgewidth=1,
                        label=alt_corr_label,
                    )

        if 'exclusion_bounds' in kwargs and kwargs['exclusion_bounds'] is not None:
            exclusion_bounds = kwargs['exclusion_bounds']
            if exclusion_bounds['type'] in ['phi', 'rho']:
                x_limits = config['limits']
                
                if exclusion_bounds['type'] == 'phi':
                    coord_step = (x_limits[1] - x_limits[0]) / corr_matrix.shape[1]
                    exclusion_left = x_limits[0] + exclusion_bounds['col_start'] * coord_step
                    exclusion_right = x_limits[0] + exclusion_bounds['col_end'] * coord_step
                    
                    ax.axvline(x=exclusion_left, color='red', linestyle='--', alpha=0.7, linewidth=1, label='Exclusion zone')
                    ax.axvline(x=exclusion_right, color='red', linestyle='--', alpha=0.7, linewidth=1)
                    
                elif exclusion_bounds['type'] == 'rho':
                    coord_step = (x_limits[1] - x_limits[0]) / corr_matrix.shape[1]
                    exclusion_left = x_limits[0] + exclusion_bounds['col_start'] * coord_step
                    exclusion_right = x_limits[0] + exclusion_bounds['col_end'] * coord_step
                    
                    ax.axvline(x=exclusion_left, color='red', linestyle='--', alpha=0.7, linewidth=1, label='Exclusion zone')
                    ax.axvline(x=exclusion_right, color='red', linestyle='--', alpha=0.7, linewidth=1)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        cbar = fig.colorbar(c)
        cbar.set_label('Correlation', fontsize=14)

        if positions.coord_system == "cylindrical":
            if positions.rec_type == "phiz":
                plt.xlabel("Azimuth Angle, $\\phi$ [$^\\circ$]", fontsize=16)
                plt.ylabel("Depth, z [m]", fontsize=16)
            elif positions.rec_type == "rhoz":
                plt.xlabel("Distance, $\\rho$ [m]", fontsize=16)
                plt.ylabel("Depth, z [m]", fontsize=16)
        else:
            plt.xlabel("Azimuth Angle, $\\phi$[$^\\circ$]", fontsize=16)
            plt.ylabel("Zenith Angle, $\\theta$[$^\\circ$]", fontsize=16)

        if positions.coord_system == "spherical":
            plt.title(
                (
                    f"St: {station_id}, run(s) {run_number}, "
                    + f"event: {event_number}, "
                    + f"ch(s): {config['channels']}\n"
                    + f"r $\\equiv$ {config['fixed_coord']}m"
                ),
                fontsize=14,
            )
        else:
            if positions.rec_type == "phiz":
                plt.title(
                    (
                        f"St: {station_id}, run(s): {run_number}, "
                        + f"event: {event_number}, "
                        + f"ch(s): {config['channels']}\n"
                        + f"$\\rho\\equiv$ {config['fixed_coord']}m"
                    ),
                    fontsize=14,
                )
            else:
                plt.title(
                    (
                        f"Station: {station_id}, run(s): {run_number}, "
                        + f"event: {event_number}, "
                        + f"ch's: {config['channels']}, "
                        + f"$\\phi\\equiv$ {config['fixed_coord']}°"
                    ),
                    fontsize=14,
                )
        
        save_dir = config['save_plots_to']
        
        minimaps = config.get('create_minimaps', False)
        if minimaps:
            try:
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                
                has_alt = ('coord0_alt' in kwargs and 'coord1_alt' in kwargs and
                          kwargs['coord0_alt'] is not None and kwargs['coord1_alt'] is not None and 
                          not np.isnan(kwargs['coord0_alt']) and not np.isnan(kwargs['coord1_alt']))
                
                if has_alt:
                    coord0_alt = kwargs['coord0_alt']
                    coord1_alt = kwargs['coord1_alt']
                    
                    if 'alt_indices' in kwargs and kwargs['alt_indices'] is not None:
                        alt_idx0, alt_idx1 = kwargs['alt_indices']
                        alt_x_center = x_midpoints[alt_idx0]
                        alt_y_center = y_midpoints[alt_idx1]
                    else:
                        if positions.coord_system == "cylindrical" and positions.rec_type == "phiz":
                            alt_x_center = coord0_alt / units.deg
                            alt_y_center = coord1_alt / units.m
                        elif positions.coord_system == "cylindrical" and positions.rec_type == "rhoz":
                            alt_x_center = coord0_alt / units.m
                            alt_y_center = coord1_alt / units.m
                        else:
                            alt_x_center = coord0_alt / units.deg
                            alt_y_center = coord1_alt / units.deg
                    
                    primary_x = max_corr_x
                    alt_x = alt_x_center
                    
                    if primary_x <= alt_x:
                        left_point_x, left_point_y = primary_x, max_corr_y
                        right_point_x, right_point_y = alt_x, alt_y_center
                        left_is_primary = True
                    else:
                        left_point_x, left_point_y = alt_x, alt_y_center
                        right_point_x, right_point_y = primary_x, max_corr_y
                        left_is_primary = False
                    
                    zoom_width = 20   # 20 degrees around peak
                    zoom_height = 10  # 10 meters around peak
                    
                    left_zoom_x_min = left_point_x - zoom_width/2
                    left_zoom_x_max = left_point_x + zoom_width/2
                    left_zoom_y_min = left_point_y - zoom_height/2
                    left_zoom_y_max = left_point_y + zoom_height/2
                    
                    left_x_start_idx = np.searchsorted(x_edges, left_zoom_x_min)
                    left_x_end_idx = np.searchsorted(x_edges, left_zoom_x_max, side='right')
                    left_y_start_idx = np.searchsorted(y_edges, left_zoom_y_min)
                    left_y_end_idx = np.searchsorted(y_edges, left_zoom_y_max, side='right')
                    
                    left_zoom_region = corr_matrix[left_y_start_idx:left_y_end_idx, left_x_start_idx:left_x_end_idx]
                    if left_zoom_region.size > 0:
                        left_vmin = np.nanmin(left_zoom_region)
                        left_vmax = np.nanmax(left_zoom_region)

                        if np.abs(left_vmax - left_vmin) < 0.001:
                            left_vmin -= 0.01
                            left_vmax += 0.01
                    else:
                        left_vmin, left_vmax = np.nanmin(corr_matrix), np.nanmax(corr_matrix)
                    
                    inset_ax_left = inset_axes(ax, width="20%", height="20%", loc='lower left', borderpad=3)
                    mycmap = plt.get_cmap("RdBu_r")
                    inset_ax_left.pcolormesh(
                        x_edges, y_edges, corr_matrix,
                        cmap=mycmap, vmin=left_vmin, vmax=left_vmax, rasterized=True
                    )
                    
                    if left_is_primary:
                        inset_ax_left.plot(left_point_x, left_point_y, "o", markersize=8, color="lime")
                    else:
                        inset_ax_left.plot(left_point_x, left_point_y, "o", markersize=8, color="lime", fillstyle="none", markeredgewidth=2)
                    
                    inset_ax_left.set_xlim(left_zoom_x_min, left_zoom_x_max)
                    inset_ax_left.set_ylim(left_zoom_y_min, left_zoom_y_max)
                    inset_ax_left.tick_params(labelsize=8)
                    inset_ax_left.set_xlabel('')
                    inset_ax_left.set_ylabel('')
                    inset_ax_left.set_title('')
                    for spine in inset_ax_left.spines.values():
                        spine.set_edgecolor('white')
                        spine.set_linewidth(1)
                    
                    right_zoom_x_min = right_point_x - zoom_width/2
                    right_zoom_x_max = right_point_x + zoom_width/2
                    right_zoom_y_min = right_point_y - zoom_height/2
                    right_zoom_y_max = right_point_y + zoom_height/2
                    
                    right_x_start_idx = np.searchsorted(x_edges, right_zoom_x_min)
                    right_x_end_idx = np.searchsorted(x_edges, right_zoom_x_max, side='right')
                    right_y_start_idx = np.searchsorted(y_edges, right_zoom_y_min)
                    right_y_end_idx = np.searchsorted(y_edges, right_zoom_y_max, side='right')
                    
                    right_zoom_region = corr_matrix[right_y_start_idx:right_y_end_idx, right_x_start_idx:right_x_end_idx]
                    if right_zoom_region.size > 0:
                        right_vmin = np.nanmin(right_zoom_region)
                        right_vmax = np.nanmax(right_zoom_region)

                        if np.abs(right_vmax - right_vmin) < 0.001:
                            right_vmin -= 0.01
                            right_vmax += 0.01
                    else:
                        right_vmin, right_vmax = np.nanmin(corr_matrix), np.nanmax(corr_matrix)
                    
                    inset_ax_right = inset_axes(ax, width="20%", height="20%", loc='lower right', borderpad=3)
                    inset_ax_right.pcolormesh(
                        x_edges, y_edges, corr_matrix,
                        cmap=mycmap, vmin=right_vmin, vmax=right_vmax, rasterized=True
                    )
                    
                    if left_is_primary:
                        inset_ax_right.plot(right_point_x, right_point_y, "o", markersize=8, color="lime", fillstyle="none", markeredgewidth=2)
                    else:
                        inset_ax_right.plot(right_point_x, right_point_y, "o", markersize=8, color="lime")
                    
                    inset_ax_right.set_xlim(right_zoom_x_min, right_zoom_x_max)
                    inset_ax_right.set_ylim(right_zoom_y_min, right_zoom_y_max)
                    inset_ax_right.tick_params(labelsize=8)
                    inset_ax_right.set_xlabel('')
                    inset_ax_right.set_ylabel('')
                    inset_ax_right.set_title('')
                    for spine in inset_ax_right.spines.values():
                        spine.set_edgecolor('white')
                        spine.set_linewidth(1)
                    
                else:
                    zoom_width = 25   # 25 degrees around peak
                    zoom_height = 12  # 12 meters around peak
                    
                    zoom_x_min = max_corr_x - zoom_width/2
                    zoom_x_max = max_corr_x + zoom_width/2
                    zoom_y_min = max_corr_y - zoom_height/2
                    zoom_y_max = max_corr_y + zoom_height/2
                    
                    single_x_start_idx = np.searchsorted(x_edges, zoom_x_min)
                    single_x_end_idx = np.searchsorted(x_edges, zoom_x_max, side='right')
                    single_y_start_idx = np.searchsorted(y_edges, zoom_y_min)
                    single_y_end_idx = np.searchsorted(y_edges, zoom_y_max, side='right')
                    
                    single_zoom_region = corr_matrix[single_y_start_idx:single_y_end_idx, single_x_start_idx:single_x_end_idx]
                    if single_zoom_region.size > 0:
                        single_vmin = np.nanmin(single_zoom_region)
                        single_vmax = np.nanmax(single_zoom_region)

                        if np.abs(single_vmax - single_vmin) < 0.001:
                            single_vmin -= 0.01
                            single_vmax += 0.01
                    else:
                        single_vmin, single_vmax = np.nanmin(corr_matrix), np.nanmax(corr_matrix)
                    
                    inset_ax = inset_axes(ax, width="25%", height="25%", loc='lower right', borderpad=3)
                    
                    mycmap = plt.get_cmap("RdBu_r")
                    inset_ax.pcolormesh(
                        x_edges, y_edges, corr_matrix,
                        cmap=mycmap, vmin=single_vmin, vmax=single_vmax, rasterized=True
                    )
                    inset_ax.plot(max_corr_x, max_corr_y, "o", markersize=8, color="lime")
                    inset_ax.set_xlim(zoom_x_min, zoom_x_max)
                    inset_ax.set_ylim(zoom_y_min, zoom_y_max)
                    inset_ax.tick_params(labelsize=8)

                    inset_ax.set_xlabel('')
                    inset_ax.set_ylabel('')
                    inset_ax.set_title('')
                    for spine in inset_ax.spines.values():
                        spine.set_edgecolor('white')
                        spine.set_linewidth(1.5)
                                    
            except Exception as e:
                import traceback
                traceback.print_exc()
        
        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        
        if show_actual_pulser or show_rec_pulser or ('coord0_alt' in kwargs and 'coord1_alt' in kwargs) or ('exclusion_bounds' in kwargs and kwargs['exclusion_bounds'] is not None):
            ax.legend()
        else:
            ax.legend()
        if file_name is not None:
            plt.savefig(save_dir + file_name)
        else:
            plt.savefig(
                save_dir + f'station{station_id}_run{run_number}_evt{event_number}.png'
            )
            print(f"\nSaved figure to {save_dir + f'station{station_id}_run{run_number}_evt{event_number}.png'}")


class StationInfo:
    def __init__(self, station, det):
        self.station = station
        self.det = det

class Positions:

    def __init__(self, station_info, limits, step_sizes, coord_system, fixed_coord, rec_type):
        self.station_info = station_info
        self.limits = limits
        self.step_sizes = step_sizes
        self.coord_system = coord_system
        self.fixed_coord = fixed_coord
        self.rec_type = rec_type
        self.ant_locs = self.get_ant_locs()
        self.coord0_vec, self.coord1_vec = self.generate_coord_arrays()

    def get_ant_locs(self):
        """
        Get antenna locations in ENU coordinates relative to the station origin (power string position at surface).
        
        Returns:
            dict: ch -> np.array([x, y, z]) in ENU coordinates
        """
        all_ch_ids = range(0, 24)
        all_xyz_ant_locs = {}
        for ch in all_ch_ids:
            all_xyz_ant_locs[ch] = np.array(
                self.station_info.det.get_relative_position(
                    int(self.station_info.station), int(ch)
                )
            )

        return all_xyz_ant_locs 

    def generate_coord_arrays(self):
        """
        From specified limits and step size, create arrays for each coordinate in its proper units.
        
        Returns:
            tuple: (coord0_vec, coord1_vec) with proper units applied
        """
        left, right, bottom, top = self.limits

        coord0_vec = np.arange(
            left, right + self.step_sizes[0], self.step_sizes[0]
        )
        coord1_vec = np.arange(
            bottom, top - self.step_sizes[1], -self.step_sizes[1]
        )

        if coord0_vec[-1] < right:
            coord0_vec = np.append(coord0_vec, right)
        if coord1_vec[-1] < top:
            coord1_vec = np.append(coord1_vec, top)

        if self.coord_system == "cylindrical":
            if self.rec_type == "phiz":
                coord0_vec = [coord0 * units.deg for coord0 in coord0_vec]
                coord1_vec = [coord1 * units.m for coord1 in coord1_vec]
            elif self.rec_type == "rhoz":
                coord0_vec = [coord0 * units.m for coord0 in coord0_vec]
                coord1_vec = [coord1 * units.m for coord1 in coord1_vec]
        elif self.coord_system == "spherical":
            coord0_vec = [coord0 * units.deg for coord0 in coord0_vec]
            coord1_vec = [coord1 * units.deg for coord1 in coord1_vec]

        return coord0_vec, coord1_vec

    def get_coord_grids(self):
        """
        Generate coordinate grids using pre-computed coordinate arrays.
        
        Returns:
            tuple: Coordinate grids in the appropriate order for the coordinate system
        """
        if self.coord_system == "cylindrical":
            if self.rec_type == "phiz":
                phi_grid, z_grid = np.meshgrid(self.coord0_vec, self.coord1_vec)
                rho_grid = np.full_like(phi_grid, self.fixed_coord)
                return rho_grid, phi_grid, z_grid
            elif self.rec_type == "rhoz":
                rho_grid, z_grid = np.meshgrid(self.coord0_vec, self.coord1_vec)
                phi_grid = np.full_like(rho_grid, self.fixed_coord)
                return rho_grid, phi_grid, z_grid
            else:
                raise ValueError(f"Invalid rec_type: {self.rec_type}")

        elif self.coord_system == "spherical":
            phi_grid, theta_grid = np.meshgrid(self.coord0_vec, self.coord1_vec)
            r_grid = np.full_like(phi_grid, self.fixed_coord)
            return r_grid, phi_grid, theta_grid

        else:
            raise ValueError(f"Unsupported coordinate system: {self.coord_system}")


    def get_source_enu_matrix(self):
        """
        Returns a matrix of potential source locations in ENU coords relative to power string position at surface.
        
        Returns:
            tuple: (src_xyz_loc_matrix, coord_arrays, grid_tuple)
        """
        coord_grids = self.get_coord_grids()
        x_grid, y_grid, z_grid = self.get_enu_coordinates(coord_grids)

        src_xyz_loc_matrix = np.stack((x_grid, y_grid, z_grid), axis=-1)

        return src_xyz_loc_matrix, [self.coord0_vec, self.coord1_vec], coord_grids

    def get_enu_coordinates(self, coords):
        """
        Converts coordinate grids to ENU coordinates relative to power string position at surface.
        """

        if self.coord_system == "cylindrical":
            rhos = coords[0]
            phis = coords[1]
            zs = coords[2]

            eastings = rhos * np.cos(phis)
            northings = rhos * np.sin(phis)
            elevations = zs

        elif self.coord_system == "spherical":
            rs = coords[0]
            phis = coords[1]
            thetas = coords[2]

            eastings = rs * np.sin(thetas) * np.cos(phis)
            northings = rs * np.sin(thetas) * np.sin(phis)
            elevations = rs * np.cos(thetas)

        return eastings, northings, elevations

    def get_rec_locs_from_corr_map(self, corr_matrix):
        """
        Extract the best (highest correlation value) reconstruction coordinates from the correlation matrix.
        """
        rec_pulser_loc0_idx, rec_pulser_loc1_idx = get_max_val_indices(
            corr_matrix
        )
        coord0_best = self.coord0_vec[rec_pulser_loc0_idx]
        coord1_best = self.coord1_vec[rec_pulser_loc1_idx]

        return coord0_best, coord1_best

def get_max_val_indices(matrix):
    
    max_val = np.nanmax(matrix)
    max_locs = np.argwhere(matrix == max_val)
    best_row_index, best_col_index = max_locs[np.random.choice(len(max_locs))]

    return best_col_index, best_row_index


def find_alternate_coordinate(corr_matrix, positions, exclude_radius_deg=5.0):
    """
    Find an alternate reconstruction coordinate by excluding the region around the primary maximum.
    
    Parameters
    ----------
    corr_matrix : array
        Correlation matrix
    positions : Positions object
        Position object containing coordinate information
    exclude_radius_deg : float
        Radius in degrees around primary maximum to exclude when finding alternate
        
    Returns
    -------
    tuple : (coord0_alt, coord1_alt, alt_indices, exclusion_bounds) or (None, None, None, None) if no alternate found
    """
    if positions.coord_system != "cylindrical" or positions.rec_type != "phiz":
        # Only implement for cylindrical phiz system for now
        return None, None, None, None
    
    primary_max_idx = np.unravel_index(np.nanargmax(corr_matrix), corr_matrix.shape)
    primary_coord0_idx = primary_max_idx[1]
    primary_coord1_idx = primary_max_idx[0]
    
    coord0_vec = positions.coord0_vec
    coord1_vec = positions.coord1_vec
    
    coord0_deg = np.array([c / units.deg for c in coord0_vec])
    coord1_m = np.array([c / units.m for c in coord1_vec])
    
    primary_azimuth = coord0_deg[primary_coord0_idx]
    primary_depth = coord1_m[primary_coord1_idx]
    
    mask = np.ones_like(corr_matrix, dtype=bool)
    
    exclusion_col_start = None
    exclusion_col_end = None
    
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            azimuth = coord0_deg[j]
            depth = coord1_m[i]
            
            azimuth_diff = abs(azimuth - primary_azimuth)
            azimuth_diff = min(azimuth_diff, 360 - azimuth_diff)
            
            if azimuth_diff <= exclude_radius_deg:
                mask[i, j] = False
                if exclusion_col_start is None or j < exclusion_col_start:
                    exclusion_col_start = j
                if exclusion_col_end is None or j > exclusion_col_end:
                    exclusion_col_end = j
    
    masked_corr = np.where(mask, corr_matrix, np.nan)
    
    if np.all(np.isnan(masked_corr)):
        return None, None, None, None
    
    alternate_max_idx = np.unravel_index(np.nanargmax(masked_corr), masked_corr.shape)
    alternate_coord0_idx = alternate_max_idx[1]
    alternate_coord1_idx = alternate_max_idx[0]
    
    coord0_alt = coord0_vec[alternate_coord0_idx]
    coord1_alt = coord1_vec[alternate_coord1_idx]
    
    exclusion_bounds = None
    if exclusion_col_start is not None and exclusion_col_end is not None:
        exclusion_bounds = {
            'type': 'phi',
            'col_start': exclusion_col_start,
            'col_end': exclusion_col_end
        }
    
    return coord0_alt, coord1_alt, (alternate_coord0_idx, alternate_coord1_idx), exclusion_bounds


def save_results_to_hdf5(results, filepath, config):
    """
    Save reconstruction results to HDF5 file.
    
    Parameters
    ----------
    results : list of dict
        List of result dictionaries with keys like 'run', 'event', 'max_corr', etc.
    filepath : str
        Path to output HDF5 file
    config : dict
        Configuration dictionary to save as attributes
    """
    if not results:
        print("No results to save.")
        return
    
    columns = sorted(results[0].keys())
    dtype = [(col, np.array([row[col] for row in results]).dtype) for col in columns]
    structured_array = np.array(
        [tuple(row[col] for col in columns) for row in results],
        dtype=dtype
    )
    
    structured_array.sort(order=["runNum", "eventNum"])
    
    with h5py.File(filepath, 'w') as f:
        dset = f.create_dataset("reconstruction", data=structured_array, compression="gzip")
        
        for key, val in config.items():
            try:
                if isinstance(val, (list, tuple)):
                    f.attrs[key] = str(val)
                elif isinstance(val, bool):
                    f.attrs[key] = int(val)
                else:
                    f.attrs[key] = val
            except (TypeError, ValueError):
                f.attrs[key] = str(val)
    
    print(f"Saved {len(results)} reconstruction results to {filepath}")
