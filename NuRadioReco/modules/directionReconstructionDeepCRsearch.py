import matplotlib.pyplot as plt
import numpy as np
import itertools
import argparse
from datetime import datetime
import os
import yaml
import logging

from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units
from NuRadioReco.detector.detector import Detector
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.modules.channelResampler import channelResampler
from NuRadioReco.modules.channelBandPassFilter import channelBandPassFilter
from NuRadioReco.modules.channelCWNotchFilter import channelCWNotchFilter
from NuRadioReco.modules.channelSinewaveSubtraction import sinewave_subtraction
from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
from NuRadioReco.detector import detector

from scipy.signal import correlate, correlation_lags
from scipy.interpolate import RegularGridInterpolator
logger = logging.getLogger("NuRadioReco.modules.directionReconstractionDeepCRsearch")

"""
This module provides a class for directional reconstruction by fitting time delays between channels to predifined time delay maps.
Usage requires pre-calculated time delay tables for each channel and configuration file specifying reconstruction parameters.
"""

def load_rz_interpolator(table_filename):
    file = np.load(table_filename)
    travel_time_table = file['data']
    r_range = file['r_range_vals']
    z_range = file['z_range_vals']

    interpolator = RegularGridInterpolator(
        (r_range, z_range), travel_time_table, method="linear", bounds_error=False, fill_value=-np.inf
    )
    return interpolator

def get_cable_delays(det, station_id, channels):
    cable_delays = {}
    for channel in channels:
        cable_delay = det.get_cable_delay(station_id, channel)
        cable_delays[channel] = cable_delay / units.ns
    return cable_delays

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

class directionReconstructionDeepCRsearch():
    """
    This module performs directional reconstruction by fitting time delays between channels to pre-defined time delay maps.
    """
    def __init__(self):
        self.begin()

    def begin(self):
        pass

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

        station_info = StationInfo(station.get_id(), config)

        positions = Positions(
            station_info,
            config['limits'],
            config['step_sizes'],
            config['coord_system'],
            config['fixed_coord'],
            config['rec_type'],
        )

        src_posn_enu_matrix, _, grid_tuple = positions.get_source_enu_matrix()

        if config.get('apply_cable_delays', True):
            cable_delays = get_cable_delays(det, station.get_id(), config['channels'])
        else:
            cable_delays = {ch: 0.0 for ch in config['channels']}

        delay_matrices = get_t_delay_matrices(
            station, config, src_posn_enu_matrix, positions.ant_locs, cable_delays
        )

        volt_arrays = []
        time_arrays = []
        
        for ch in config['channels']:
            channel = station.get_channel(ch)
            trace = channel.get_trace()
            
            if config.get('apply_cw_removal', False):
                trace = sinewave_subtraction(trace, saved_noise_freqs=[])
            
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

        # this needs to be fixed/better generalized
        if config["coord_system"] == "cylindrical":
            num_rows_to_10m = int(np.ceil(10 / abs(config['step_sizes'][1])))
            surface_corr= self.get_surf_corr(corr_matrix, num_rows_to_10m)
        elif config["coord_system"] == "spherical":
            num_rows_to_10m = 10
            surface_corr= self.get_surf_corr(corr_matrix, num_rows_to_10m)
        else:
            surface_corr = -1.0

        station.set_parameter(stnp.rec_max_correlation, max_corr)
        station.set_parameter(stnp.rec_surf_corr, surface_corr)

        if config["coord_system"] == "cylindrical":
            if config["rec_type"] == "phiz":
                station.set_parameter(stnp.rec_azimuth, rec_coord0)  # φ (azimuth)
                station.set_parameter(stnp.rec_z, rec_coord1)        # z (depth)
            elif config["rec_type"] == "rhoz":
                station.set_parameter(stnp.rec_azimuth, rec_coord0)  # ρ (radius) - stored in azimuth param
                station.set_parameter(stnp.rec_z, rec_coord1)        # z (depth)
        elif config["coord_system"] == "spherical":
            station.set_parameter(stnp.rec_azimuth, rec_coord0)      # φ (azimuth)
            station.set_parameter(stnp.rec_zenith, rec_coord1)       # θ (zenith)
    
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
        if show_actual_pulser or show_rec_pulser:
            plt.legend()
        if file_name is not None:
            plt.savefig(save_dir + file_name)
        else:
            plt.savefig(
                save_dir + f'station{station_id}_run{run_number}evt{event_number}.png'
            )
            print(f"\nSaved figure to {save_dir + f'station{station_id}_run{run_number}_evt{event_number}.png'}")


class StationInfo:
    def __init__(self, station, config):
        self.station = station
        self.det = Detector(json_filename=config['detector_json'])
        self.det.update(datetime(2022, 10, 1))

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
        From specified limits and step size, create arrays for each coordinate
        in its proper units.
        
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
        Returns a matrix of potential source locations in ENU coords
        relative to power string position at surface.
        
        Returns:
            tuple: (src_xyz_loc_matrix, coord_arrays, grid_tuple)
        """
        # Get coordinate grids and convert to ENU coordinates
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

def corr_index_from_t(time_delay, times):
    """
    Finds the index in the correlation corresponding to the given time delay.
    """
    center = len(times)
    dt = times[1] - times[0]
    
    time_delay_clean = np.nan_to_num(time_delay, nan=0.0)

    index = np.rint(time_delay_clean / dt + (center - 1)).astype(int)
    
    return index


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="%(prog)s", usage="reconstruction test")
    parser.add_argument("--station", type=int, default=21)
    parser.add_argument("--run", type=int, default=240430)
    parser.add_argument("--event", type=int, default=4241)
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--save_plots", default=True, help="Will save correlation map plot if true")
    parser.add_argument("--verbose", default=True, help="If true, will print out reconstruction results")

    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    readRNOGData = readRNOGData()  

    root_dirs = os.path.join(f"{config['data_dir']}", f"station{args.station}", f"run{args.run}", "combined.root")
    
    # processing that includes block offset corrections and voltage conversion
    readRNOGData.begin(root_dirs, mattak_kwargs={'backend': 'uproot'})
    
    print(f"Event {args.event}, Station {args.station}, Run {args.run}")

    channel_resampler = channelResampler()
    channel_resampler.begin()
    
    channel_bandpass_filter = channelBandPassFilter()
    channel_bandpass_filter.begin()

    reco = directionReconstructionDeepCRsearch()
    reco.begin()

    event=readRNOGData.get_event(args.run, args.event)
    station=event.get_station()

    det = detector.Detector(source="rnog_mongo")
    det.update(datetime(2022, 10, 1))
    
    if config.get('apply_upsampling', False):
        print("  - Upsampling to 5 GHz")
        channel_resampler.run(event, station, det, sampling_rate=5 * units.GHz)
    
    if config.get('apply_bandpass', False):
        print("  - Applying bandpass filter: 0.1-0.6 GHz")
        channel_bandpass_filter.run(event, station, det, 
            passband=[0.1 * units.GHz, 0.6 * units.GHz],
            filter_type='butter', order=10)

    reco.run(event, station, det, config, corr_map=args.save_plots)
    
    if args.verbose:
        print(f"\n=== Reconstruction Results ===")
        print(f"Max correlation: {station.get_parameter(stnp.rec_max_correlation):.3f}")
        print(f"Surface correlation: {station.get_parameter(stnp.rec_surf_corr):.3f}")
        
        if config["coord_system"] == "cylindrical":
            if config["rec_type"] == "phiz":
                print(f"Reconstructed azimuth (φ): {station.get_parameter(stnp.rec_azimuth)/units.deg:.1f}°")
                print(f"Reconstructed depth (z): {station.get_parameter(stnp.rec_z)/units.m:.1f} m")
                print(f"Fixed radius (ρ): {config['fixed_coord']} m")
            elif config["rec_type"] == "rhoz":
                print(f"Reconstructed radius (ρ): {station.get_parameter(stnp.rec_azimuth)/units.m:.1f} m")  # Note: stored in azimuth param for rhoz
                print(f"Reconstructed depth (z): {station.get_parameter(stnp.rec_z)/units.m:.1f} m")
                print(f"Fixed azimuth (φ): {config['fixed_coord']}°")
        elif config["coord_system"] == "spherical":
            print(f"Reconstructed azimuth (φ): {station.get_parameter(stnp.rec_azimuth)/units.deg:.1f}°")
            print(f"Reconstructed zenith (θ): {station.get_parameter(stnp.rec_zenith)/units.deg:.1f}°")
            print(f"Fixed radius (r): {config['fixed_coord']} m")
        
        print(f"Coordinate system: {config['coord_system']}")
        print(f"Reconstruction type: {config['rec_type']}")
        print("===============================\n")

    reco.end()
    readRNOGData.end()
