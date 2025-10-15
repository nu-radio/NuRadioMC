import numpy as np
import itertools
import os
import yaml
import logging
from functools import lru_cache

from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units
from NuRadioReco.utilities import caching_utilities
from NuRadioReco.detector.detector import Detector
from NuRadioReco.framework.parameters import stationParameters as stnp

from scipy.signal import correlate, correlation_lags
from scipy.interpolate import RegularGridInterpolator
from NuRadioReco.utilities.interferometry_io_utilities import save_correlation_map

logger = logging.getLogger("NuRadioReco.modules.interferometricDirectionReconstruction")

"""
This module provides a class for directional reconstruction by fitting time delays between channels to predifined time delay maps.
Usage requires pre-calculated time delay tables for each channel and configuration file specifying reconstruction parameters.
"""


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
    
    @staticmethod
    @lru_cache(maxsize=128)
    def _load_rz_interpolator(table_filename):
        """Load R-Z interpolator from time delay table file."""
        file = np.load(table_filename)
        travel_time_table = file['data']
        r_range = file['r_range_vals']
        z_range = file['z_range_vals']

        interpolator = RegularGridInterpolator(
            (r_range, z_range), travel_time_table, method="linear", bounds_error=False, fill_value=-np.inf
        )
        return interpolator
    
    @staticmethod
    def _get_t_delay_matrices(station, config, src_posn_enu_matrix, ant_locs, cable_delays):
        """Compute time delay matrices for all channel pairs."""
        ch_pairs = list(itertools.combinations(config['channels'], 2))
        time_delay_matrices = []

        data_dir = config['time_delay_tables'] 
        outdir = data_dir + f"station{station}/"

        interpolators = {}
        for ch in set(itertools.chain(*ch_pairs)):
            table_file = f"{outdir}ch{ch}_rz_table_rel_ant.npz"
            interpolators[ch] = interferometricDirectionReconstruction._load_rz_interpolator(table_file)

        for ch1, ch2 in ch_pairs:
            pos1 = ant_locs[ch1]
            pos2 = ant_locs[ch2]
            interp1 = interpolators[ch1]
            interp2 = interpolators[ch2]

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
    
    @staticmethod
    def _correlator(times, v_array_pairs, delay_matrices, apply_hann_window=False, use_hilbert=False):
        """Compute correlation between channel pairs given time delay matrices."""
        volt_corrs = []
        time_lags_list = []
        
        channels = list(range(len(times)))
        channel_pairs = list(itertools.combinations(channels, 2))
        
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
                v1_processed = np.abs(hilbert(v1))
                v2_processed = np.abs(hilbert(v2))
            else:
                v1_processed = v1
                v2_processed = v2
            
            if apply_hann_window:
                window1 = np.hanning(len(v1_processed))
                window2 = np.hanning(len(v2_processed))
                v1_processed = v1_processed * window1
                v2_processed = v2_processed * window2
            
            v1_processed = (v1_processed - np.mean(v1_processed)) / np.std(v1_processed)
            v2_processed = (v2_processed - np.mean(v2_processed)) / np.std(v2_processed)
            
            corr = correlate(v1_processed, v2_processed, mode='full', method='auto')
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

    def begin(self, station_id=None, config=None, det=None):
        """
        Initialize the module, optionally preloading delay matrices and positions from disk cache.
        
        Parameters
        ----------
        station_id : int, optional
            Station ID to preload delay matrices for. If provided along with config,
            will load existing cache files into memory to avoid repeated disk I/O.
        config : dict or str, optional
            Configuration to use for generating the cache key. Required if station_id is set.
        det : detector.Detector, optional
            Detector object for precomputing positions and delay matrices.
        """
        if station_id is not None and config is not None:
            if isinstance(config, str):
                with open(config, "r") as f:
                    config = yaml.safe_load(f)
            
            if det is not None:
                self._precompute_station_data(station_id, config, det)

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

        self.ant_locs = self._get_ant_locs(station_id, det)

        if config.get('apply_cable_delays', True):
            if station_id not in self._cable_delay_cache:
                self._cable_delay_cache[station_id] = {
                    ch: det.get_cable_delay(station_id, ch) / units.ns
                    for ch in range(24)
                }
            cable_delays = {ch: self._cable_delay_cache[station_id][ch] for ch in channels}
        else:
            cable_delays = {ch: 0.0 for ch in channels}

        cache_key = caching_utilities.generate_cache_key(
            station_id,
            tuple(sorted(channels)),
            tuple(limits),
            tuple(step_sizes),
            coord_system,
            fixed_coord,
            rec_type,
            tuple(sorted((ch, round(delay, 6)) for ch, delay in cable_delays.items()))
        )
        
        if cache_key not in self._delay_matrix_cache:
            cache_path = caching_utilities.get_cache_path(
                "delay_matrices",
                f"station{station_id}_delays_{cache_key}.pkl"
            )
            delay_matrices = caching_utilities.load_from_cache(cache_path)
            
            if delay_matrices is None:
                src_posn_enu_matrix, [coord0_vec, coord1_vec], grid_tuple = self._get_source_enu_matrix(
                    station_id, det, limits, step_sizes, coord_system, fixed_coord, rec_type
                )
                delay_matrices = self._get_t_delay_matrices(
                    station_id, config, src_posn_enu_matrix, self.ant_locs, cable_delays
                )
                
                metadata = {
                    'cache_key': cache_key,
                    'station': station_id,
                    'channels': channels,
                    'limits': limits,
                    'coord_system': coord_system,
                    'rec_type': rec_type,
                    'fixed_coord': fixed_coord
                }
                caching_utilities.save_to_cache(delay_matrices, cache_path, metadata)
            
            self._delay_matrix_cache[cache_key] = delay_matrices

        coord0_vec, coord1_vec = self._generate_coord_arrays(limits, step_sizes, coord_system, rec_type)
        

        self._station_delay_matrices[station_id] = {
            'coord0_vec': coord0_vec,
            'coord1_vec': coord1_vec,
            'coord_system': coord_system,
            'rec_type': rec_type,
            'delay_matrices': self._delay_matrix_cache[cache_key],
            'cache_key': cache_key
        }

    @register_run()
    def run(self, evt, station, det, config, save_maps=False, save_maps_to=None):
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
        
        # Retrieve cached data directly by station_id
        cached_data = self._station_delay_matrices[station_id]
        coord0_vec = cached_data['coord0_vec']
        coord1_vec = cached_data['coord1_vec']
        coord_system = cached_data['coord_system']
        rec_type = cached_data['rec_type']
        delay_matrices = cached_data['delay_matrices']

        channels = config['channels']
        volt_arrays = []
        time_arrays = []
        
        for ch in channels:
            channel = station.get_channel(ch)
            trace = channel.get_trace()

            volt_arrays.append(trace)
            time_arrays.append(channel.get_times())

        v_array_pairs = list(itertools.combinations(volt_arrays, 2))

        corr_matrix, max_corr = self._correlator(
            time_arrays, v_array_pairs, delay_matrices,
            apply_hann_window=config.get('apply_hann_window', False),
            use_hilbert=config.get('use_hilbert_envelope', False)
        )
        
        rec_coord0, rec_coord1 = self._get_rec_locs_from_corr_map(
            corr_matrix, coord0_vec, coord1_vec
        )

        coord0_alt, coord1_alt, alt_indices, exclusion_bounds = None, None, None, None
        if config.get('find_alternate_reco', False):
            exclude_radius = config.get('alternate_exclude_radius_deg', 5.0)
            result = self._find_alternate_coordinate(corr_matrix, coord0_vec, coord1_vec, coord_system, rec_type, exclude_radius)
            if result[0] is not None and result[1] is not None:
                coord0_alt, coord1_alt, alt_indices, exclusion_bounds = result
        
        # Handle correlation map visualization and saving
        if save_maps:
            map_kwargs = {}
            if coord0_alt is not None and coord1_alt is not None:
                map_kwargs['coord0_alt'] = coord0_alt
                map_kwargs['coord1_alt'] = coord1_alt
                map_kwargs['alt_indices'] = alt_indices
            if exclusion_bounds is not None:
                map_kwargs['exclusion_bounds'] = exclusion_bounds
            
            if save_maps_to is not None:
                save_dir = save_maps_to
            else:
                save_dir = config.get('save_maps_to', config.get('save_plots_to', './correlation_maps/'))
            
            # Create a simple dict to pass position info to save function
            position_dict = {
                'coord0_vec': coord0_vec,
                'coord1_vec': coord1_vec,
                'coord_system': coord_system,
                'rec_type': rec_type
            }
            save_correlation_map(corr_matrix, position_dict, evt=evt, config=config, save_dir=save_dir, **map_kwargs)

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
    
    @staticmethod
    def _get_max_val_indices(matrix):
        """Get indices of maximum value in matrix (with random selection if multiple maxima)."""
        max_val = np.nanmax(matrix)
        max_locs = np.argwhere(matrix == max_val)
        best_row_index, best_col_index = max_locs[np.random.choice(len(max_locs))]
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
            Reconstruction type ('phiz', 'rhoz', etc.)
        exclude_radius_deg : float
            Radius in degrees around primary maximum to exclude when finding alternate
            
        Returns
        -------
        tuple : (coord0_alt, coord1_alt, alt_indices, exclusion_bounds) or (None, None, None, None) if no alternate found
        """
        if coord_system != "cylindrical" or rec_type != "phiz":
            return None, None, None, None
        
        primary_max_idx = np.unravel_index(np.nanargmax(corr_matrix), corr_matrix.shape)
        primary_coord0_idx = primary_max_idx[1]
        primary_coord1_idx = primary_max_idx[0]
        
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
    

    def _get_ant_locs(self, station_id, det):
        """Get antenna locations in ENU coordinates for a station."""
        all_ch_ids = range(0, 24)
        all_xyz_ant_locs = {}
        for ch in all_ch_ids:
            all_xyz_ant_locs[ch] = np.array(
                det.get_relative_position(int(station_id), int(ch))
            )
        return all_xyz_ant_locs

    def _generate_coord_arrays(self, limits, step_sizes, coord_system, rec_type):
        """Generate coordinate arrays with proper units."""
        left, right, bottom, top = limits
        
        buffer = 0.5 # buffer R [m] used to avoid weirdness that happens near 0
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
        
        ch_1_loc = self.ant_locs[1]
        ch_2_loc = self.ant_locs[2]
        center_of_PA = (ch_1_loc + ch_2_loc) / 2.0
                
        """Convert coordinate grids to ENU coordinates."""
        if coord_system == "cylindrical":
            rhos = coords[0]
            phis = coords[1]
            zs = coords[2]
            eastings = rhos * np.cos(phis)
            northings = rhos * np.sin(phis)
            elevations = zs
        elif coord_system == "spherical":
            rs = coords[0]
            phis = coords[1]
            thetas = coords[2]
            eastings = rs * np.sin(thetas) * np.cos(phis) + center_of_PA[0]
            northings = rs * np.sin(thetas) * np.sin(phis) + center_of_PA[1]
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
