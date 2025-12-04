import numpy as np
import itertools
import os
import sys
import yaml
import logging
from functools import lru_cache
from scipy.signal import windows
import time

from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units
from NuRadioReco.detector.detector import Detector
from NuRadioReco.framework.parameters import stationParameters as stnp

from scipy.signal import correlate, correlation_lags, hilbert
from scipy.interpolate import RegularGridInterpolator
from NuRadioReco.utilities.interferometry_io_utilities import save_correlation_map
from NuRadioReco.utilities.caching_utilities import (
    generate_cache_key, get_cache_path, load_from_cache, save_to_cache
)

logger = logging.getLogger("NuRadioReco.modules.interferometricDirectionReconstruction")

# Try to import Numba for accelerated interpolation
USE_NUMBA = False
try:
    from numba import njit, prange
    USE_NUMBA = True
    logger.info("Numba available - will use JIT-compiled interpolation for correlation interpolation speedup")
except ImportError:
    logger.info("Numba not available - using standard numpy interpolation for correlation")

logger = logging.getLogger("NuRadioReco.modules.interferometricDirectionReconstruction")

# Numba-accelerated uniform grid interpolation (if available)
if USE_NUMBA:
    @njit(parallel=True, fastmath=True)
    def _interp_uniform_numba(y, dt, offset, x):
        """
        Fast uniform grid interpolation using Numba JIT compilation.
        
        Compiled to native code with parallel execution across x values.
        Significantly faster than np.interp for large arrays.
        
        Parameters
        ----------
        y : array
            Correlation values (uniform grid)
        dt : float
            Time step between correlation samples
        offset : float
            Time value of first correlation sample
        x : array
            Query points (delay times to interpolate at)
        
        Returns
        -------
        array : Interpolated values at x positions
        """
        M = y.shape[0]
        n = x.shape[0]
        out = np.empty(n, dtype=np.float64)
        
        for i in prange(n):
            kf = (x[i] - offset) / dt
            k = int(np.floor(kf))
            
            if k < 0 or k >= M - 1:
                out[i] = np.nan
            else:
                alpha = kf - k
                out[i] = y[k] + (y[k+1] - y[k]) * alpha
        
        return out

# Try to import C++ extension for fast delay matrix computation
# The extension is located in NuRadioReco/examples/RNOG/interferometric_reco_ex/
# We add that directory to sys.path temporarily to import it
USE_CPP_EXTENSION = False
try:
    # Get the path to the C++ extension relative to NuRadioReco package
    # This works regardless of where NuRadioMC is installed
    import NuRadioReco
    nuradoreco_path = os.path.dirname(NuRadioReco.__file__)
    cpp_extension_path = os.path.join(nuradoreco_path, 'examples', 'RNOG', 'interferometric_reco_ex')
    
    # Temporarily add to path, import, then remove (clean approach)
    if cpp_extension_path not in sys.path:
        sys.path.insert(0, cpp_extension_path)
        try:
            from fast_delay_matrices import compute_delay_matrices as compute_delay_matrices_cpp
            USE_CPP_EXTENSION = True
            logger.info("C++ extension loaded successfully - will use fast C++ implementation for building of time delay matrices")
        finally:
            # Remove from path to keep it clean
            sys.path.remove(cpp_extension_path)
    else:
        from fast_delay_matrices import compute_delay_matrices as compute_delay_matrices_cpp
        USE_CPP_EXTENSION = True
        logger.info("C++ extension loaded successfully - will use fast C++ implementation for building of time delay matrices")
except (ImportError, OSError) as e:
    USE_CPP_EXTENSION = False
    logger.info("C++ extension not available - will use Python implementation for building time delay matrices. Error: %s", str(e))


#USE_CPP_EXTENSION=False
"""
This module provides a class for directional reconstruction by fitting time delays between channels to predifined time delay maps.
Usage requires pre-calculated time delay tables for each channel and configuration file specifying reconstruction parameters.
"""


class interferometricDirectionReconstruction():
    """
    This module performs directional reconstruction by fitting time delays between channels to pre-defined time delay maps.
    """
    def __init__(self):
        # Cache coordinate vectors with config-based keys for multi-stage reconstruction
        self._coord_vec_cache = {}  # Key: (coord_system, rec_type, tuple(limits), tuple(step_sizes))
        self.begin()
    
    @staticmethod
    @lru_cache(maxsize=128)
    def _load_rz_interpolator(table_filename, interpolation_method):
        """Load R-Z interpolator from time delay table file."""
        file = np.load(table_filename)
        travel_time_table = file['data']
        r_range = file['r_range_vals']
        z_range = file['z_range_vals']
        # pick the map with fastest rays out of (0,1,2,3), there 0-2 solution types and 3 is fastest rays
        interpolator = RegularGridInterpolator(
            (r_range, z_range), travel_time_table[3], method=interpolation_method, bounds_error=False, fill_value=-np.inf
        )
        return interpolator

    def _get_t_delay_matrices(self, station, config, src_posn_enu_matrix, ant_locs, interpolators):
        """
        Compute time delay matrices using interpolators (standard tables).
        Uses C++ extension if available, otherwise Python.
        
        Parameters
        ----------
        station : int
            Station ID
        config : dict
            Configuration dictionary
        src_posn_enu_matrix : array
            Source position matrix in ENU coordinates
        ant_locs : dict
            Antenna locations
        interpolators : dict
            Dictionary mapping channel_id -> interpolator
        """
        # Try C++ extension first (fastest)
        if USE_CPP_EXTENSION:
            try:
                channels = np.array(config['channels'], dtype=np.int32)
                return compute_delay_matrices_cpp(
                    channels, src_posn_enu_matrix, ant_locs, interpolators
                )
            except Exception as e:
                logger.info("C++ extension failed (%s), falling back to Python", str(e))

        ch_pairs = list(itertools.combinations(config['channels'], 2))
        
        # Pre-allocate buffers outside the channel loop for reuse
        grid_shape = src_posn_enu_matrix.shape[:2]
        flat_size = grid_shape[0] * grid_shape[1]
        
        # Reusable buffers for interpolator input and distance calculation
        coords_rel = np.empty((flat_size, 2), dtype=np.float64)
        dx = np.empty(grid_shape, dtype=np.float64)
        dy = np.empty(grid_shape, dtype=np.float64)
        
        # Pre-compute travel times for all unique channels (vectorized)
        travel_times = {}
        z_grid = src_posn_enu_matrix[:, :, 2]  # Z is same for all channels
        xy_positions = src_posn_enu_matrix[:, :, :2]  # Extract XY once
        
        for ch in config['channels']:
            pos = ant_locs[ch]
            interp = interpolators[ch]
            
            # Compute rho distance efficiently using pre-allocated buffers
            # Manual sqrt is faster than np.linalg.norm
            np.subtract(xy_positions[:, :, 0], pos[0], out=dx)
            np.subtract(xy_positions[:, :, 1], pos[1], out=dy)
            np.multiply(dx, dx, out=dx)  # dx^2 in place
            np.multiply(dy, dy, out=dy)  # dy^2 in place
            np.add(dx, dy, out=dx)       # dx now holds dx^2 + dy^2
            np.sqrt(dx, out=dx)          # dx now holds rho
            
            # Prepare flattened (N, 2) array for interpolator
            coords_rel[:, 0] = dx.ravel()
            coords_rel[:, 1] = z_grid.ravel()
            
            # Interpolate and reshape back
            travel_times[ch] = interp(coords_rel).reshape(grid_shape)
        
        # Compute pairwise differences
        time_delay_matrices = []
        for ch1, ch2 in ch_pairs:
            time_delay_matrix = travel_times[ch1] - travel_times[ch2]
            time_delay_matrices.append(time_delay_matrix)

        return time_delay_matrices
    
    @staticmethod
    def _correlator(times, v_array_pairs, delay_matrices, apply_hann_window=False, use_hilbert=False):
        """
        Compute correlation between channel pairs given time delay matrices.
        """
        logger.debug("Entering _correlator: %d channel pairs, %d delay matrices", len(v_array_pairs), len(delay_matrices))
        
        n_channels = len(times)
        channel_pairs = list(itertools.combinations(range(n_channels), 2))
        n_pairs = len(channel_pairs)
        
        # Pre-compute time sampling rates for all channels
        dts = np.array([t[1] - t[0] if len(t) > 1 else 1.0 for t in times])
        
        # Get shape of delay matrices (all should be same shape)
        matrix_shape = delay_matrices[0].shape
        
        logger.debug("n_pairs=%d, matrix_shape=%s, trace_length=%d", 
                    n_pairs, matrix_shape, len(v_array_pairs[0][0]) if v_array_pairs else 0)
        
        # Pre-allocate the full output array (all pair correlation matrices)
        pair_corr_matrices_array = np.full((n_pairs, *matrix_shape), np.nan, dtype=np.float64)
        
        # Pre-compute correlation_lags and overlap_norm once (all traces same length)
        len_v_array = len(v_array_pairs[0][0])
        lags_template = correlation_lags(len_v_array, len_v_array, mode="full")
        overlap_norm = correlate(np.ones(len_v_array), np.ones(len_v_array), mode='full')
        
        # Single pass: Compute correlation AND interpolate immediately
        for pair_idx in range(n_pairs):
            v1, v2 = v_array_pairs[pair_idx]
            
            ch1_idx, ch2_idx = channel_pairs[pair_idx]
            t1, t2 = times[ch1_idx], times[ch2_idx]
            dt = min(dts[ch1_idx], dts[ch2_idx])
            
            # if use_hilbert:
            #     v1 = np.abs(hilbert(v1))
            #     v2 = np.abs(hilbert(v2))
            
            # Vectorized normalization
            mean1 = v1.mean()
            mean2 = v2.mean()
            std1 = v1.std()
            std2 = v2.std()
            
            if std1 == 0 or std2 == 0:
                logger.warning("Zero std encountered in channel pair %d: std1=%s, std2=%s", pair_idx, std1, std2)
                v1_normalized = v1 - mean1
                v2_normalized = v2 - mean2
            else:
                v1_normalized = (v1 - mean1) / std1
                v2_normalized = (v2 - mean2) / std2
            
            # Compute correlation
            corr = correlate(v1_normalized, v2_normalized, mode='full', method='auto')
            
            if use_hilbert:
                corr = np.abs(hilbert(corr))
            
            # Normalize by overlap
            corr_normalized = corr / overlap_norm
            
            if apply_hann_window:
                hann_window = windows.hann(len(corr_normalized))
                corr_normalized *= hann_window
            
            # Interpolate using fastest available method
            time_delay = delay_matrices[pair_idx]
            valid_mask = ~np.isnan(time_delay)
            
            if np.any(valid_mask):
                if USE_NUMBA:
                    # Numba path: Use JIT-compiled parallel interpolation
                    # Compute offset for uniform grid: time_lags[0] = -(M // 2) * dt + (t1[0] - t2[0])
                    M = len(corr_normalized)
                    offset = -(M // 2) * dt + (t1[0] - t2[0])
                    
                    # Extract valid delays and interpolate with Numba
                    x = time_delay[valid_mask].ravel().astype(np.float64)
                    vals = _interp_uniform_numba(
                        corr_normalized.astype(np.float64), 
                        float(dt), 
                        float(offset), 
                        x
                    )
                    pair_corr_matrices_array[pair_idx][valid_mask] = vals
                else:
                    # Standard numpy path: Use np.interp
                    time_lags = lags_template * dt + t1[0] - t2[0]
                    valid_delays = time_delay[valid_mask]
                    interp_corr = np.interp(valid_delays, time_lags, corr_normalized)
                    pair_corr_matrices_array[pair_idx][valid_mask] = interp_corr
        
        # Aggregation: Use nansum divided by total number of pairs
        mean_corr_matrix = np.nansum(pair_corr_matrices_array, axis=0) / n_pairs
        
        if np.all(np.isnan(mean_corr_matrix)):
            logger.warning("Mean correlation matrix is all NaN (no valid pair contributions)")
            max_corr = np.nan
        else:
            max_corr = np.nanmax(mean_corr_matrix)
        
        logger.debug("Exiting _correlator: mean_corr_matrix shape=%s, max_corr=%s", mean_corr_matrix.shape, str(max_corr))
        
        # Convert back to list for compatibility with original return signature
        pair_corr_matrices = [pair_corr_matrices_array[i] for i in range(n_pairs)]
        
        return pair_corr_matrices, mean_corr_matrix, max_corr

    def begin(self, station_id=None, config=None, det=None):
        """
        Initialize the module, optionally preloading interpolators.
        
        Parameters
        ----------
        station_id : int, optional
            Station ID to preload data for
        config : dict or str, optional
            Configuration dictionary or path to YAML file
        det : detector.Detector, optional
            Detector object (required for antenna locations)
        """
        
        if station_id is not None and config is not None:

            if isinstance(config, str):
                with open(config, "r") as f:
                    config = yaml.safe_load(f)
            
            self._preload_tables(station_id, config)
            
            if not hasattr(self, 'ant_locs') or self.ant_locs is None:
                try:
                    self.ant_locs = self._get_ant_locs(station_id, det)
                except Exception as e:
                    logger.error("Could not compute ant_locs for station %s: %s", station_id, e)
                    raise
    
    def _preload_tables(self, station_id, config):
        """
        Pre-load travel time table interpolators with caching.
        
        Parameters
        ----------
        station_id : int
            Station ID
        config : dict
            Configuration dictionary containing:
            - 'channels': list of channel IDs
            - 'time_delay_tables': path to tables directory
            - 'interp_method': interpolation method (default: 'linear')
        """
        channels_tuple = tuple(sorted(config['channels']))
        table_path = config['time_delay_tables']
        interp_method = config.get('interp_method', 'linear')
        
        # Cache key depends on: station, channels, table path, and method
        cache_key = generate_cache_key(
            station_id, 
            channels_tuple, 
            table_path,
            interp_method
        )
        
        cache_filename = f"interpolators_st{station_id}_{cache_key}.pkl"
        cache_filepath = get_cache_path("interferometry_interpolators", cache_filename)
        
        # Try to load from cache
        cached_interpolators = load_from_cache(cache_filepath)
        
        if cached_interpolators is not None:
            logger.info(f"Loaded interpolators from cache for station {station_id}")
            self._interpolators = cached_interpolators
            return
        
        # Cache miss - need to load and create interpolators
        print(f"Pre-loading time delay table interpolators for station {station_id}, channels {config['channels']}...", flush=True)
        
        self._interpolators = {}
        for ch in config['channels']:
            table_file = os.path.join(
                f"{config['time_delay_tables']}", 
                f"station{station_id}", 
                f"ch{ch}_rz_table_rel_ant_allRays_fixed_withFastest.npz"
            )
            self._interpolators[ch] = self._load_rz_interpolator(table_file, interp_method)
        
        print("Done!\n", flush=True)
        
        # Save to cache for future use
        metadata = {
            'station_id': station_id,
            'channels': list(channels_tuple),
            'table_path': table_path,
            'interp_method': interp_method
        }
        save_to_cache(self._interpolators, cache_filepath, metadata=metadata)
        logger.debug(f"Saved interpolators to cache: {cache_filepath}")

    @register_run()
    def run(self, evt, station, det, config, save_maps=False, save_pair_maps=False, save_maps_to=None):
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
            save_maps: bool
                If True, saves correlation map data to pickle files for later plotting.
            save_maps_to: str, optional
                Directory to save correlation maps. If None, uses config settings or defaults.
        """
        
        if isinstance(config, str):
            config = self.load_config(config)

        station_id = station.get_id()
        logger.debug("Running interferometricDirectionReconstruction for station %s, event %s", station_id, evt.get_id())
        
        # Generate coord arrays from current config (don't use cached ones - they may be from different coord system)
        coord0_vec, coord1_vec = self._generate_coord_arrays(
            config['limits'], config['step_sizes'], config['coord_system'], config['rec_type']
        )
        coord_system = config['coord_system']
        rec_type = config['rec_type']
                    
        # Compute source ENU matrix and delay matrices
        src_posn_enu_matrix, _, _ = self._get_source_enu_matrix(
            station_id, det, config['limits'], config['step_sizes'], 
            config['coord_system'], config['fixed_coord'], config['rec_type'],
        )
        
        # Compute time delay matrices
        delay_matrices = self._get_t_delay_matrices(
            station_id, config, src_posn_enu_matrix, self.ant_locs, self._interpolators
        )

        channels = config['channels']
        volt_arrays = []
        time_arrays = []
        
        for ch in channels:
            channel = station.get_channel(ch)
            volt_arrays.append(channel.get_trace())
            time_arrays.append(channel.get_times())
            
        v_array_pairs = list(itertools.combinations(volt_arrays, 2))
        logger.debug("Prepared %d voltage arrays and %d pairs for correlation", len(volt_arrays), len(v_array_pairs))

        pair_corr_matrices, corr_matrix, max_corr = self._correlator(
            time_arrays, v_array_pairs, delay_matrices,
            apply_hann_window=config.get('apply_hann_window', False),
            use_hilbert=config.get('use_hilbert_envelope', False)
        )
        logger.debug("Correlation matrix shape: %s, max_corr: %s", np.shape(corr_matrix), str(max_corr))
        
        rec_coord0, rec_coord1 = self._get_rec_locs_from_corr_map(
            corr_matrix, coord0_vec, coord1_vec
        )
        logger.debug(f"Coordinate extraction details:")
        logger.debug(f"  Corr matrix shape: {corr_matrix.shape}")
        logger.debug(f"  Max correlation value: {np.nanmax(corr_matrix):.4f}")
        logger.debug(f"  Max index in matrix: {np.unravel_index(np.nanargmax(corr_matrix), corr_matrix.shape)}")
        logger.debug(f"  coord0_vec length: {len(coord0_vec)}, range: [{coord0_vec[0]}, {coord0_vec[-1]}]")
        logger.debug(f"  coord1_vec length: {len(coord1_vec)}, range: [{coord1_vec[0]}, {coord1_vec[-1]}]")
        logger.debug(f"  Extracted rec_coord0: {rec_coord0}")
        logger.debug(f"  Extracted rec_coord1: {rec_coord1}")
        logger.debug(f"  Coord system: {coord_system}, rec_type: {rec_type}")
        
        coord0_alt, coord1_alt, alt_indices, exclusion_bounds = None, None, None, None
        if config.get('find_alternate_reco', False):
            exclude_radius = config.get('alternate_exclude_radius_deg', 5.0)
            result = self._find_alternate_coordinate(corr_matrix, coord0_vec, coord1_vec, coord_system, rec_type, exclude_radius)
            if result[0] is not None and result[1] is not None:
                coord0_alt, coord1_alt, alt_indices, exclusion_bounds = result
                
        # Handle correlation map visualization and saving
        if save_maps or save_pair_maps:
            # Determine save directory: use provided path, else config, else default
            if save_maps_to is not None:
                save_dir = save_maps_to
            else:
                save_dir = config.get('save_maps_to', config.get('save_plots_to', './correlation_maps/'))
            
            map_kwargs = {}
            if coord0_alt is not None and coord1_alt is not None:
                map_kwargs['coord0_alt'] = coord0_alt
                map_kwargs['coord1_alt'] = coord1_alt
                map_kwargs['alt_indices'] = alt_indices
            if exclusion_bounds is not None:
                map_kwargs['exclusion_bounds'] = exclusion_bounds
            
            # Create a simple dict to pass position info to save function
            position_dict = {
                'coord0_vec': coord0_vec,
                'coord1_vec': coord1_vec,
                'coord_system': coord_system,
                'rec_type': rec_type
            }
            # Provide the reconstructed coordinates and max corr so the plotter doesn't need to recompute
            map_kwargs['rec_coord0'] = rec_coord0
            map_kwargs['rec_coord1'] = rec_coord1
            map_kwargs['rec_max_corr'] = max_corr
            print(f"config rec type: {config['rec_type']}")
            print(f"rec type: {rec_type}")
            full_corr_map_save_path = save_correlation_map(corr_matrix, position_dict, evt=evt, config=config, save_dir=save_dir, **map_kwargs)
            print(f"full corr map save path: {full_corr_map_save_path}")
            logger.debug("Saved full correlation map to %s (event %s)", save_dir, evt.get_id())

            logger.debug("save_pair_maps flag = %s", save_pair_maps)
            if save_pair_maps:
                pair_save_dir = os.path.join(save_dir, "pairwise_maps")
                logger.info("Saving pair map data to: %s", pair_save_dir)
                os.makedirs(pair_save_dir, exist_ok=True)
                for idx, pair_corr in enumerate(pair_corr_matrices):
                    ch1, ch2 = list(itertools.combinations(channels, 2))[idx]
                    pair_map_kwargs = map_kwargs.copy()
                    pair_map_kwargs['filename_suffix'] = f"_ch{ch1}-ch{ch2}"
                    pair_map_kwargs['title_suffix'] = f" (Ch {ch1} & Ch {ch2})"
                    pair_map_kwargs['pair_channels'] = [ch1, ch2]  # Store channel pair info in map data
                    # Include recon info for pair maps as well (use same rec coords/max from combined map)
                    pair_map_kwargs['rec_coord0'] = rec_coord0
                    pair_map_kwargs['rec_coord1'] = rec_coord1
                    pair_map_kwargs['rec_max_corr'] = max_corr
                    # Also extract THIS pair's individual maximum correlation point
                    pair_rec_coord0, pair_rec_coord1 = self._get_rec_locs_from_corr_map(
                        pair_corr, coord0_vec, coord1_vec
                    )
                    pair_map_kwargs['pair_rec_coord0'] = pair_rec_coord0
                    pair_map_kwargs['pair_rec_coord1'] = pair_rec_coord1
                    pair_map_kwargs['pair_rec_max_corr'] = np.nanmax(pair_corr)
                    _ = save_correlation_map(pair_corr, position_dict, evt=evt, config=config, save_dir=pair_save_dir, **pair_map_kwargs)
                    logger.info("Saved pair map for channels %s-%s to %s", ch1, ch2, pair_save_dir)
        else:
            full_corr_map_save_path = None
        
        # Optional: Save delay matrices for debugging comparison with multitable reconstruction
        if config.get('save_delay_matrices', False):
            import pickle
            delay_save_path = config.get('delay_matrices_save_path', './debug_delay_matrices_singletable.pkl')
            debug_data = {
                'delay_matrices': delay_matrices,
                'coord0_vec': coord0_vec,
                'coord1_vec': coord1_vec,
                'config': config,
                'station_id': station_id,
                'event_id': evt.get_id(),
                'run_number': evt.get_run_number(),
                'channels': channels,
            }
            with open(delay_save_path, 'wb') as f:
                pickle.dump(debug_data, f)
            logger.info("Saved delay matrices to %s for debugging (singletable method)", delay_save_path)
        
        step_sizes = config['step_sizes']
        
        if coord_system == "cylindrical":
            num_rows_to_10m = int(np.ceil(10 / abs(step_sizes[1])))
            surface_corr= self.get_surf_corr(corr_matrix, num_rows_to_10m)
            station.set_parameter(stnp.rec_surf_corr, surface_corr)
        else:
            station.set_parameter(stnp.rec_surf_corr, np.nan)

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
                        
        return full_corr_map_save_path
    
    @staticmethod
    def _get_max_val_indices(matrix):
        """Get indices of maximum value in matrix (with random selection if multiple maxima)."""
        max_val = np.nanmax(matrix)
        max_locs = np.argwhere(matrix == max_val)
        best_row_index, best_col_index = max_locs[np.random.choice(len(max_locs))]
        logger.debug(f"  _get_max_val_indices: max_val={max_val:.4f}, returning col={best_col_index}, row={best_row_index}")
        return best_col_index, best_row_index
    
    @staticmethod
    def _find_alternate_coordinate(corr_matrix, coord0_vec, coord1_vec, coord_system, rec_type, exclude_radius_deg=5.0):
        """
        Find an alternate reconstruction coordinate by excluding the region around the primary maximum.
        
        Parameters
        ----------
        corr_matrix : array
            Correlation matrix
        coord0_vec : array
            First coordinate vector
        coord1_vec : array
            Second coordinate vector
        coord_system : str
            Coordinate system ('cylindrical' or 'spherical')
        rec_type : str
            Reconstruction type ('phiz', 'rhoz', 'phitheta', etc.)
        exclude_radius_deg : float
            Radius in degrees around primary maximum to exclude when finding alternate
            
        Returns
        -------
        tuple : (coord0_alt, coord1_alt, alt_indices, exclusion_bounds) or (None, None, None, None) if no alternate found
        """
        # Only works for reconstructions where coord0 is azimuth (φ)
        # This includes: cylindrical phiz and spherical phitheta
        if not ((coord_system == "cylindrical" and rec_type == "phiz") or 
                (coord_system == "spherical" and rec_type == "phitheta")):
            return None, None, None, None
        
        primary_max_idx = np.unravel_index(np.nanargmax(corr_matrix), corr_matrix.shape)
        primary_coord0_idx = primary_max_idx[1]
        primary_coord1_idx = primary_max_idx[0]
        
        # coord0 is always azimuth (φ) in degrees for both phiz and phitheta
        coord0_deg = np.array([c / units.deg for c in coord0_vec])
        
        # coord1 is depth (z) for phiz or zenith angle (θ) for phitheta
        if coord_system == "cylindrical" and rec_type == "phiz":
            coord1_values = np.array([c / units.m for c in coord1_vec])
        elif coord_system == "spherical" and rec_type == "phitheta":
            coord1_values = np.array([c / units.deg for c in coord1_vec])
        
        primary_azimuth = coord0_deg[primary_coord0_idx]
        primary_coord1 = coord1_values[primary_coord1_idx]
        
        mask = np.ones_like(corr_matrix, dtype=bool)
        
        # Store actual azimuth bounds (in degrees), not column indices
        exclusion_azimuth_min = primary_azimuth - exclude_radius_deg
        exclusion_azimuth_max = primary_azimuth + exclude_radius_deg
        
        exclusion_col_start = None
        exclusion_col_end = None
                
        for i in range(corr_matrix.shape[0]):
            for j in range(corr_matrix.shape[1]):
                azimuth = coord0_deg[j]
                coord1_val = coord1_values[i]
                
                # Exclude based on azimuth difference (wraps around at 360°)
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
                
        # Store exclusion bounds in actual azimuth degrees (wrapping handled in plotting)
        exclusion_bounds = {
            'type': 'phi',
            'azimuth_center': primary_azimuth,
            'azimuth_min': exclusion_azimuth_min,
            'azimuth_max': exclusion_azimuth_max,
            'radius_deg': exclude_radius_deg
        }
                
        return coord0_alt, coord1_alt, (alternate_coord0_idx, alternate_coord1_idx), exclusion_bounds
    

    def _get_ant_locs(self, station_id, det):
        """
        Get antenna locations with absolute Z coordinates.
        
        For time delay table lookups, we need:
        - x, y: Can be relative to station center (only used for computing R distances)
        - z: MUST be absolute depth (same reference frame as the Z values in time delay tables)
        
        The time delay tables are indexed by (R, Z_absolute), where:
        - R is computed from antenna x,y position
        - Z_absolute is the absolute depth below surface
        """
        all_ch_ids = range(0, 24)
        all_xyz_ant_locs = {}
        
        # Get station's absolute position to extract absolute Z offset
        station_abs_pos = det.get_absolute_position(int(station_id))
        station_abs_z = station_abs_pos[2]
        
        for ch in all_ch_ids:
            ant_rel_pos = np.array(det.get_relative_position(int(station_id), int(ch)))
            # Keep x,y relative (only used for computing R distances)
            # But convert z to absolute by adding station's absolute z
            all_xyz_ant_locs[ch] = ant_rel_pos.copy()
            all_xyz_ant_locs[ch][2] = ant_rel_pos[2] + station_abs_z
             
        return all_xyz_ant_locs

    def _generate_coord_arrays(self, limits, step_sizes, coord_system, rec_type):
        """
        Generate coordinate arrays with proper units and cache them.
        
        Uses a cache keyed by (coord_system, rec_type, limits, step_sizes) to avoid 
        regenerating coordinate vectors for repeated reconstructions with the same config.
        This is especially important for multi-stage reconstructions that reuse the same
        stage configurations across many events.
        """
        # Create cache key from config parameters
        cache_key = (coord_system, rec_type, tuple(limits), tuple(step_sizes))
        
        # Check if we've already generated these coordinate vectors
        if cache_key in self._coord_vec_cache:
            logger.debug(f"Using cached coordinate vectors for {coord_system}/{rec_type}")
            return self._coord_vec_cache[cache_key]
        
        # Generate new coordinate vectors
        logger.debug(f"Generating new coordinate vectors for {coord_system}/{rec_type}")
        left, right, bottom, top = limits
        
        buffer = 0.0 # buffer R [m] used to avoid weirdness that happens near 0
        if coord_system == "cylindrical" and left < buffer:
            coord0_vec = np.arange(buffer, right + step_sizes[0], step_sizes[0])
        else:
            coord0_vec = np.arange(left, right + step_sizes[0], step_sizes[0])
        
        coord1_vec = np.arange(bottom, top + step_sizes[1], step_sizes[1])
            
        if coord_system == "cylindrical":
            if rec_type == "phiz":
                coord0_vec = [coord0 * units.deg for coord0 in coord0_vec]
                coord1_vec = [coord1 * units.m for coord1 in coord1_vec]
            elif rec_type == "rhoz":
                coord0_vec = [coord0 * units.m for coord0 in coord0_vec]
                coord1_vec = [coord1 * units.m for coord1 in coord1_vec]
        elif coord_system == "spherical":
            coord0_vec = [coord0 * units.deg for coord0 in coord0_vec]
            coord1_vec = [coord1 * units.deg for coord1 in coord1_vec]
        
        # Cache for future use
        self._coord_vec_cache[cache_key] = (coord0_vec, coord1_vec)
        
        return coord0_vec, coord1_vec

    def _get_coord_grids(self, coord0_vec, coord1_vec, coord_system, rec_type, fixed_coord):
        """Generate coordinate grids."""
        if coord_system == "cylindrical":
            if rec_type == "phiz":
                phi_grid, z_grid = np.meshgrid(coord0_vec, coord1_vec)
                rho_grid = np.full_like(phi_grid, fixed_coord)
                return rho_grid, phi_grid, z_grid
            elif rec_type == "rhoz":
                rho_grid, z_grid = np.meshgrid(coord0_vec, coord1_vec)
                phi_grid = np.full_like(rho_grid, fixed_coord)
                return rho_grid, phi_grid, z_grid
            else:
                raise ValueError(f"Invalid rec_type: {rec_type}")
        elif coord_system == "spherical":
            phi_grid, theta_grid = np.meshgrid(coord0_vec, coord1_vec)
            r_grid = np.full_like(phi_grid, fixed_coord)
            return r_grid, phi_grid, theta_grid
        else:
            raise ValueError(f"Unsupported coordinate system: {coord_system}")

    def _get_enu_coordinates(self, coords, coord_system):
        """
        Convert coordinate grids to ENU coordinates.
        
        For cylindrical coordinates (rho, phi, z):
        - x, y are relative to PA center (arbitrary reference, only used for R calculation)
        - z is absolute depth (used directly for table lookup)
        
        For spherical coordinates (r, phi, theta):
        - x, y are computed relative to PA center
        - z is computed as absolute depth (PA absolute z + radial component)
        """
        
        ch_1_loc = self.ant_locs[1]
        ch_2_loc = self.ant_locs[2]
        center_of_PA = (ch_1_loc + ch_2_loc) / 2.0
                        
        if coord_system == "cylindrical":
            rhos = coords[0]
            phis = coords[1]
            zs = coords[2]
            # x,y relative to PA (arbitrary, only for computing R)
            eastings = rhos * np.cos(phis)
            northings = rhos * np.sin(phis)
            # z is already absolute from config (e.g., -200m to 0m below surface)
            elevations = zs
            
        elif coord_system == "spherical":
            rs = coords[0]
            phis = coords[1]
            thetas = coords[2]
            # x,y computed relative to PA center (arbitrary reference)
            eastings = rs * np.sin(thetas) * np.cos(phis) + center_of_PA[0]
            northings = rs * np.sin(thetas) * np.sin(phis) + center_of_PA[1]
            # z is absolute: PA absolute z + radial component in z direction
            elevations = rs * np.cos(thetas) + center_of_PA[2]
            
        return eastings, northings, elevations

    def _get_source_enu_matrix(self, station_id, det, limits, step_sizes, coord_system, fixed_coord, rec_type):
        """Get matrix of potential source locations in ENU coordinates."""
        coord0_vec, coord1_vec = self._generate_coord_arrays(limits, step_sizes, coord_system, rec_type)
        coord_grids = self._get_coord_grids(coord0_vec, coord1_vec, coord_system, rec_type, fixed_coord)
            
        x_grid, y_grid, z_grid = self._get_enu_coordinates(coord_grids, coord_system)
        src_xyz_loc_matrix = np.stack((x_grid, y_grid, z_grid), axis=-1)
        return src_xyz_loc_matrix, [coord0_vec, coord1_vec], coord_grids

    def _get_rec_locs_from_corr_map(self, corr_matrix, coord0_vec, coord1_vec):
        """Extract best reconstruction coordinates from correlation matrix."""
        rec_pulser_loc0_idx, rec_pulser_loc1_idx = self._get_max_val_indices(corr_matrix)
        logger.debug(f"  _get_rec_locs_from_corr_map: max indices = ({rec_pulser_loc0_idx}, {rec_pulser_loc1_idx})")
        logger.debug(f"  _get_rec_locs_from_corr_map: coord0_vec[{rec_pulser_loc0_idx}] = {coord0_vec[rec_pulser_loc0_idx]}")
        logger.debug(f"  _get_rec_locs_from_corr_map: coord1_vec[{rec_pulser_loc1_idx}] = {coord1_vec[rec_pulser_loc1_idx]}")
        coord0_best = coord0_vec[rec_pulser_loc0_idx]
        coord1_best = coord1_vec[rec_pulser_loc1_idx]
        return coord0_best, coord1_best
    
    def end(self):
        pass

    def load_config(self, config_file):
        with open(config_file, "r") as f:
            return yaml.safe_load(f)

    def get_surf_corr(self, corr_map, num_rows_for_10m):
        surf_corr = np.nanmax(corr_map[:num_rows_for_10m])
        return surf_corr
