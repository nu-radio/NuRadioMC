import matplotlib.pyplot as plt
import numpy as np
import itertools
import argparse
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
from NuRadioReco.modules.channelResampler import channelResampler
from NuRadioReco.modules.channelBandPassFilter import channelBandPassFilter
from NuRadioReco.modules.channelSinewaveSubtraction import channelSinewaveSubtraction
from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
from NuRadioReco.modules.io.eventReader import eventReader
from NuRadioReco.modules.io.eventWriter import eventWriter
from NuRadioReco.detector import detector

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
        tuple(sorted((ch, round(delay, 6)) for ch, delay in cable_delays.items()))  # Round to avoid float precision issues
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
    outdir = data_dir + f"station{station.get_id()}/"

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

        rzs1 = np.linalg.norm(src_posn_enu_matrix[:, :, :2] - pos1[:2], axis=2)
        zs1 = src_posn_enu_matrix[:, :, 2]
        rzs2 = np.linalg.norm(src_posn_enu_matrix[:, :, :2] - pos2[:2], axis=2)
        zs2 = src_posn_enu_matrix[:, :, 2]

        coords1 = np.stack((rzs1, zs1), axis=-1)
        coords2 = np.stack((rzs2, zs2), axis=-1)

        t1 = interp1(coords1)
        t2 = interp2(coords2)
        
        cable_delay_diff = cable_delays[ch1] - cable_delays[ch2]
        time_delay_matrix = t1 - t2 + cable_delay_diff

        time_delay_matrices.append(time_delay_matrix)

    return time_delay_matrices

def correlator(times, v_array_pairs, delay_matrices):

    volt_corrs = []
    time_lags_list = []
    
    channels = list(range(len(times)))
    channel_pairs = list(itertools.combinations(channels, 2))
    
    for pair_idx, (v1, v2) in enumerate(v_array_pairs):
        ch1_idx, ch2_idx = channel_pairs[pair_idx]
        t1, t2 = times[ch1_idx], times[ch2_idx]
        
        dt1 = t1[1] - t1[0] if len(t1) > 1 else 1.0
        dt2 = t2[1] - t2[0] if len(t2) > 1 else 1.0
        dt = min(dt1, dt2)
        
        corr = correlate(v1, v2, mode='full', method='auto')
        norm_factor = (np.sum(v1**2) * np.sum(v2**2))**0.5
        corr_normalized = corr / norm_factor
        volt_corrs.append(corr_normalized)
        
        lags = correlation_lags(len(v1), len(v2), mode="full")
        time_lags = lags * dt
        time_lags_list.append(time_lags)

    pair_corr_matrices = []

    for pair_idx, (volt_corr, time_lags, time_delay) in enumerate(zip(volt_corrs, time_lags_list, delay_matrices)):
        valid_mask = ~np.isnan(time_delay)
        
        pair_corr_matrix = np.zeros_like(time_delay)
        
        if np.any(valid_mask):
            valid_delays = time_delay[valid_mask].flatten()
            interp_corr = np.interp(valid_delays, time_lags, volt_corr)
            pair_corr_matrix[valid_mask] = interp_corr.reshape(np.sum(valid_mask))
        
        pair_corr_matrices.append(pair_corr_matrix)    

    mean_corr_matrix = np.mean(pair_corr_matrices, axis=0)
    max_corr = np.max(mean_corr_matrix)

    return mean_corr_matrix, max_corr

class interferometricDirectionReconstruction():
    """
    This module performs directional reconstruction by fitting time delays between channels to pre-defined time delay maps.
    """
    def __init__(self):
        self._cable_delay_cache = {}
        self._delay_matrix_cache = {}
        self.begin()

    def begin(self, preload_cache_for_station=None, config=None):
        """
        Initialize the module, optionally preloading delay matrices from disk cache.
        
        Parameters
        ----------
        preload_cache_for_station : int, optional
            Station ID to preload delay matrices for. If provided along with config,
            will load existing cache files into memory to avoid repeated disk I/O.
        config : dict or str, optional
            Configuration to use for generating the cache key. Required if preload_cache_for_station is set.
        """
        if preload_cache_for_station is not None and config is not None:
            if isinstance(config, str):
                with open(config, "r") as f:
                    config = yaml.safe_load(f)
            
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

        coord_system = config['coord_system']
        rec_type = config['rec_type']
        fixed_coord = config['fixed_coord']
        channels = config['channels']
        limits = config['limits']
        step_sizes = config['step_sizes']

        station_info = StationInfo(station.get_id(), det)

        positions = Positions(
            station_info,
            limits,
            step_sizes,
            coord_system,
            fixed_coord,
            rec_type,
        )

        src_posn_enu_matrix, _, grid_tuple = positions.get_source_enu_matrix()

        if config.get('apply_cable_delays', True):
            station_id = station.get_id()
            if station_id not in self._cable_delay_cache:
                self._cable_delay_cache[station_id] = {
                    ch: det.get_cable_delay(station_id, ch) / units.ns
                    for ch in range(24)
                }
            cable_delays = {ch: self._cable_delay_cache[station_id][ch] for ch in channels}
        else:
            cable_delays = {ch: 0.0 for ch in channels}

        cache_key = get_delay_matrix_cache_key(
            station.get_id(),
            channels,
            limits,
            step_sizes,
            coord_system,
            fixed_coord,
            rec_type,
            cable_delays
        )
        
        if cache_key in self._delay_matrix_cache:
            delay_matrices = self._delay_matrix_cache[cache_key]
        else:
            delay_matrices = load_delay_matrices_from_cache(cache_key, station.get_id())
            
            if delay_matrices is None:
                delay_matrices = get_t_delay_matrices(
                    station, config, src_posn_enu_matrix, positions.ant_locs, cable_delays
                )
                
                config_info = {
                    'station': station.get_id(),
                    'channels': channels,
                    'limits': limits,
                    'coord_system': coord_system,
                    'rec_type': rec_type,
                    'fixed_coord': fixed_coord
                }
                save_delay_matrices_to_cache(delay_matrices, cache_key, station.get_id(), config_info)
            
            self._delay_matrix_cache[cache_key] = delay_matrices

        volt_arrays = []
        time_arrays = []
        
        for ch in channels:
            channel = station.get_channel(ch)
            trace = channel.get_trace()
            
            if config.get('apply_waveform_scaling', False):
                if np.max(trace) != 0:
                    trace = trace / np.max(trace)
                if np.std(trace) != 0:
                    trace = trace / np.std(trace)
                trace = trace - np.mean(trace)
            
            volt_arrays.append(trace)
            time_arrays.append(channel.get_times())

        v_array_pairs = list(itertools.combinations(volt_arrays, 2))

        corr_matrix, max_corr = correlator(
            time_arrays, v_array_pairs, delay_matrices
        )
        
        if corr_map == True:
            self.plot_corr_map(corr_matrix, positions, evt=evt, config = config)

        rec_coord0, rec_coord1 = positions.get_rec_locs_from_corr_map(
            corr_matrix
        )

        if coord_system == "cylindrical":
            num_rows_to_10m = int(np.ceil(10 / abs(step_sizes[1])))
            surface_corr= self.get_surf_corr(corr_matrix, num_rows_to_10m)
        elif coord_system == "spherical":
            num_rows_to_10m = 10
            surface_corr= self.get_surf_corr(corr_matrix, num_rows_to_10m)
        else:
            surface_corr = -1.0

        station.set_parameter(stnp.rec_max_correlation, max_corr)
        station.set_parameter(stnp.rec_surf_corr, surface_corr)

        station.set_parameter(stnp.rec_coord0, rec_coord0)
        station.set_parameter(stnp.rec_coord1, rec_coord1)

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
        surf_corr = np.max(corr_map[:num_rows_for_10m])

        return surf_corr

    def plot_corr_map(
        self,
        corr_matrix, positions, 
        file_name=None,
        evt=None, 
        config=None,
        show_actual_pulser=True,
        show_rec_pulser=True):

        run_number = evt.get_run_number()
        event_number = evt.get_id()
        station = evt.get_station()
        station_id = station.get_id()

        mycmap = plt.get_cmap("RdBu_r")

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
            vmin=np.min(corr_matrix),
            vmax=np.max(corr_matrix),
            rasterized=True,
        )

        x_midpoints = (x_edges[:-1] + x_edges[1:]) / 2
        y_midpoints = (y_edges[:-1] + y_edges[1:]) / 2

        max_corr_value = np.max(corr_matrix)
        max_corr_indices = np.unravel_index(
            np.argmax(corr_matrix), corr_matrix.shape
        )
        max_corr_x = x_midpoints[max_corr_indices[1]]
        max_corr_y = y_midpoints[max_corr_indices[0]]
        
        ax.plot(
            max_corr_x,
            max_corr_y,
            "o",
            markersize=10,
            color="lime",
            label=f"Max corr: {max_corr_value:.2f}",
        )

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        fig.colorbar(c)

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
                    f"Station: {station_id}, run(s) {run_number}, "
                    + f"event: {event_number}, "
                    + f"ch's: {config['channels']}, \n"
                    + f"max_corr: {round(np.max(corr_matrix), 2)}, "
                    + f"r $\\equiv${config['fixed_coord']}m, "
                    + f"rec. loc ($\\phi$, $\\theta$): ({int(max_corr_x)}$^\\circ$, {int(config['limits'][3] - max_corr_y)}$^\\circ$)"
                ),
                fontsize=14,
            )
        else:
            if positions.rec_type == "phiz":
                plt.title(
                    (
                        f"Station: {station_id}, run(s): {run_number}, "
                        + f"event: {event_number}, "
                        + f"ch's: {config['channels']}\n"
                        + f"$\\rho\\equiv${config['fixed_coord']}m, rec. loc ($\\phi$, z): "
                        + f"({int(max_corr_x)}$^\\circ$, {int(max_corr_y)}m)"
                    ),
                    fontsize=14,
                )
            else:
                plt.title(
                    (
                        f"Station: {station_id}, run(s): {run_number}, "
                        + f"event: {event_number}, "
                        + f"ch's: {config['channels']}\n"
                        + f"rec. loc ($\\rho$, z): ({int(max_corr_x)}m, {int(max_corr_y)}m)"
                    ),
                    fontsize=16,
                )
        
        save_dir = config['save_plots_to']
        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        
        if show_actual_pulser or show_rec_pulser:
            plt.legend()
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
    
    max_val = np.max(matrix)
    max_locs = np.argwhere(matrix == max_val)
    best_row_index, best_col_index = max_locs[np.random.choice(len(max_locs))]

    return best_col_index, best_row_index


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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="%(prog)s", usage="reconstruction test")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file to hold parameters that don't very from one event to another")
    parser.add_argument("--inputfile", type=str, nargs="+", help="Path(s) to input data file(s) (ROOT or NUR). Can specify multiple files for same station.")
    parser.add_argument("--outputfile", type=str, default=None, help="Path to output file (.nur for NuRadioReco format with parameters only, .h5 for HDF5 table)")
    parser.add_argument("--events", type=int, nargs="*", default=None, help="Specific event IDs to process (optional). If not provided, processes all events")
    parser.add_argument("--save_plots", action="store_true", help="Will save correlation map plot if true and a subset of events are specified.")
    parser.add_argument("--verbose", action="store_true", help="If true, will print out reconstruction results")

    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    input_files = args.inputfile if isinstance(args.inputfile, list) else [args.inputfile]
    
    is_nur_file = input_files[0].endswith('.nur')
    
    channel_resampler = channelResampler()
    channel_resampler.begin()
    
    channel_bandpass_filter = channelBandPassFilter()
    channel_bandpass_filter.begin()
    
    cw_filter = channelSinewaveSubtraction()
    cw_filter.begin(
        save_filtered_freqs=False,
        freq_band=tuple(config.get('cw_freq_band', [0.1, 0.7]))
    )

    reco = interferometricDirectionReconstruction()
    
    preload_station_id = None
    if not is_nur_file:
        temp_reader = readRNOGData()
        temp_reader.begin(input_files[0], mattak_kwargs={'backend': 'uproot'})
        temp_event = next(temp_reader.run())
        preload_station_id = temp_event.get_station().get_id()
        temp_reader.end()
        
        reco.begin(preload_cache_for_station=preload_station_id, config=config)
    else:
        temp_reader = eventReader()
        temp_reader.begin(input_files[0], read_detector=True)
        temp_event = next(temp_reader.run())
        preload_station_id = temp_event.get_station().get_id()
        temp_reader.end()
        
        reco.begin(preload_cache_for_station=preload_station_id, config=config)
    
    writer = None
    if args.outputfile and args.outputfile.endswith('.nur'):
        writer = eventWriter()
        writer.begin(args.outputfile)
        print(f"Will save reconstruction results to NUR file: {args.outputfile}")

    events_to_process = set(args.events) if args.events is not None else None
    
    results = [] if args.outputfile and args.outputfile.endswith('.h5') else None
    if results is not None:
        print(f"Will save reconstruction results to HDF5 file: {args.outputfile}")
    
    n_processed = 0
    n_skipped = 0
    found_event_ids = [] 
    found_run_numbers = []
    
    for file_idx, input_file in enumerate(input_files, 1):
        
        if is_nur_file:
            reader = eventReader()
            reader.begin(input_file, read_detector=True)
            det = reader.get_detector()
            
            if det is None:
                print(f"Warning: No detector description in NUR file. Loading from config: {config['detector_json']}")
                det = detector.Detector(json_filename=config['detector_json'])
                det.update(datetime(2022, 10, 1))
            
            event_generator = reader.run()
        else:
            reader = readRNOGData()
            reader.begin(input_file, mattak_kwargs={'backend': 'uproot'})
            det = detector.Detector(source="rnog_mongo")
            det.update(datetime(2022, 10, 1))
            event_generator = reader.run()
        
        for event in event_generator:
            event_id = event.get_id()
            run_number = event.get_run_number()
            
            found_event_ids.append(event_id)
            found_run_numbers.append(run_number)
            
            # NOTE: NuRadioMC simulation files (.nur) can have a quirk where all events share
            # the same event_id (typically 0), but each event has a unique run_number. 
            # The run_number effectively serves as the event index for simulations.
            # For real data (.root files), event_id is the correct identifier.
            if events_to_process is not None:
                if is_nur_file:
                    # For NUR files (simulations), check run_number as the event index
                    if run_number not in events_to_process:
                        n_skipped += 1
                        continue
                else:
                    # For ROOT files (data), use event_id
                    if event_id not in events_to_process:
                        n_skipped += 1
                        continue
            
            station = event.get_station()
            station_id = station.get_id()
            
            for ch_id in [ch for ch in station.get_channel_ids() if ch not in config['channels']]:
                station.remove_channel(ch_id)
            
            if config.get('apply_upsampling', False):
                channel_resampler.run(event, station, det, sampling_rate=5 * units.GHz)
            
            if config.get('apply_bandpass', False):
                channel_bandpass_filter.run(event, station, det, 
                    passband=[0.1 * units.GHz, 0.6 * units.GHz],
                    filter_type='butter', order=10)
            
            if config.get('apply_cw_removal', False):
                peak_prominence = config.get('cw_peak_prominence', 4.0)
                cw_filter.run(event, station, det, peak_prominence=peak_prominence)

            reco.run(event, station, det, config, corr_map=args.save_plots)
            
            if writer is not None:
                mode = {
                    'Channels': False,
                    'ElectricFields': False,
                    'SimChannels': False,
                    'SimElectricFields': False
                }
                writer.run(event, det=det, mode=mode)
            
            max_corr = station.get_parameter(stnp.rec_max_correlation)
            surf_corr = station.get_parameter(stnp.rec_surf_corr)
            rec_coord0 = station.get_parameter(stnp.rec_coord0)
            rec_coord1 = station.get_parameter(stnp.rec_coord1)
            
            if results is not None:
                result_row = {
                    "runNum": run_number,
                    "eventNum": event_id,
                    "maxCorr": max_corr,
                    "surfCorr": surf_corr,
                }
                
                if config["coord_system"] == "cylindrical":
                    if config["rec_type"] == "phiz":
                        # coord0 = φ (degrees), coord1 = z (meters)
                        result_row["phi"] = rec_coord0 / units.deg
                        result_row["z"] = rec_coord1 / units.m
                    elif config["rec_type"] == "rhoz":
                        # coord0 = ρ (meters), coord1 = z (meters)
                        result_row["rho"] = rec_coord0 / units.m
                        result_row["z"] = rec_coord1 / units.m
                elif config["coord_system"] == "spherical":
                    # coord0 = φ (degrees), coord1 = θ (degrees)
                    result_row["phi"] = rec_coord0 / units.deg
                    result_row["theta"] = rec_coord1 / units.deg
                
                results.append(result_row)
            
            if args.verbose:
                print(f"\n=== Reconstruction Results ===")
                print(f"Station: {station_id}")
                print(f"Max correlation: {max_corr:.3f}")
                print(f"Surface correlation: {surf_corr:.3f}")
                
                if config["coord_system"] == "cylindrical":
                    if config["rec_type"] == "phiz":
                        # coord0 = φ (azimuth), coord1 = z (depth)
                        print(f"Reconstructed azimuth (φ): {rec_coord0/units.deg:.1f}°")
                        print(f"Reconstructed depth (z): {rec_coord1/units.m:.1f} m")
                        print(f"Fixed radius (ρ): {config['fixed_coord']} m")
                    elif config["rec_type"] == "rhoz":
                        # coord0 = ρ (radius), coord1 = z (depth)
                        print(f"Reconstructed radius (ρ): {rec_coord0/units.m:.1f} m")
                        print(f"Reconstructed depth (z): {rec_coord1/units.m:.1f} m")
                        print(f"Fixed azimuth (φ): {config['fixed_coord']}°")
                elif config["coord_system"] == "spherical":
                    # coord0 = φ (azimuth), coord1 = θ (zenith)
                    print(f"Reconstructed azimuth (φ): {rec_coord0/units.deg:.1f}°")
                    print(f"Reconstructed zenith (θ): {rec_coord1/units.deg:.1f}°")
                    print(f"Fixed radius (r): {config['fixed_coord']} m")
                
                print(f"Coordinate system: {config['coord_system']}")
                print(f"Reconstruction type: {config['rec_type']}")
                print("===============================\n")
            
            n_processed += 1
            
            if events_to_process is not None and n_processed == len(events_to_process):
                break
        
        if not is_nur_file:
            reader.end()
        
        if events_to_process is not None and n_processed == len(events_to_process):
            break

    reco.end()
    if writer is not None:
        writer.end()
    
    if args.outputfile and args.outputfile.endswith('.h5') and results:
        save_results_to_hdf5(results, args.outputfile, config)
    
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"Processed: {n_processed} events")
    if events_to_process is not None:
        print(f"Skipped: {n_skipped} events")
        if n_processed == 0:
            print(f"\nWARNING: None of the requested events were found!")
            print(f"Requested event indices: {sorted(events_to_process)}")
            if is_nur_file:
                unique_indices = sorted(set(found_run_numbers))
                print(f"Available event indices in file: {unique_indices[:20]}")
                if len(unique_indices) > 20:
                    print(f"... and {len(unique_indices) - 20} more events")
                print(f"   Example: --events {unique_indices[0]} {unique_indices[1] if len(unique_indices) > 1 else ''}")
            else:
                unique_events = sorted(set(found_event_ids))
                print(f"Available event IDs in file: {unique_events[:20]}")
                if len(unique_events) > 20:
                    print(f"... and {len(unique_events) - 20} more events")
    print(f"{'='*50}\n")
