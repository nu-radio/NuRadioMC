import matplotlib.pyplot as plt
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units
import NuRadioReco.detector.detector
from NuRadioReco.detector.detector import Detector
import uproot
import numpy as np
import itertools
import argparse
from datetime import datetime
import sys, os
from scipy import signal
import yaml
from scipy.interpolate import RegularGridInterpolator
# from NuRadioReco.modules.RNO_G.channelBlockOffsetFitter import fit_block_offsets
import pandas as pd
import copy
import json
from scipy.signal import correlate
from NuRadioReco.modules.base import module
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import channelParameters as chp
import logging
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

def get_t_delay_matrices(station, config, src_posn_enu_matrix, channels, ant_locs, use_random_delays=False, max_delay_ns=100):

    ch_pairs = list(itertools.combinations(channels, 2))
    time_delay_matrices = []

    data_dir = config['time_delay_tables'] 
    outdir = data_dir + f"station{station}/"

    if not use_random_delays:
        interpolators = {}
        for ch in set(itertools.chain(*ch_pairs)):
            table_file = f"{outdir}ch{ch}_rz_table_R1_1600Z-1600_200.npz"
            interpolators[ch] = load_rz_interpolator(table_file)

    for ch1, ch2 in ch_pairs:
        if use_random_delays:
            shape = src_posn_enu_matrix.shape[:2]
            time_delay_matrix = np.random.uniform(-max_delay_ns, max_delay_ns, size=shape)
        else:
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
            time_delay_matrix = t1 - t2

        time_delay_matrices.append(time_delay_matrix)

    return time_delay_matrices

def correlator(times, v_array_pairs, delay_matrices, debug_config=None):
    volt_corrs = [
        correlate(v1, v2, mode='full', method='auto') /
        (np.sum(v1 ** 2) * np.sum(v2 ** 2)) ** 0.5
        for v1, v2 in v_array_pairs
    ]

    pair_corr_matrices = []
    all_indices = [] if debug_config else None

    for time_array, volt_corr, time_delay in zip(times, volt_corrs, delay_matrices):
        indices = corr_index_from_t(time_delay, time_array)
        if debug_config:
            all_indices.append(indices)

        corr_matrix = np.full_like(time_delay, -np.inf)
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                corr_matrix[i, j] = volt_corr[indices[i, j]]

        pair_corr_matrices.append(corr_matrix)

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

        station_info = StationInfo(config["station"], config)

        positions = Positions(
            station_info,
            config['limits'],
            config['step_sizes'],
            config['coord_system'],
            config['fixed_coord'],
            config['rec_type'],
        )


        src_posn_enu_matrix, _, grid_tuple = positions.get_source_enu_matrix(config['rec_type'])

        delay_matrices = get_t_delay_matrices(
            config['station'], config, src_posn_enu_matrix, config['channels'], positions.ant_locs, use_random_delays=False
        )


        #initialize variables to save
        corr_matrix_sum = None
        max_corr_vals = -1.0
        surface_corr = -1.0


        sampling_rate = station.get_channel(0).get_sampling_rate() 

        # apply cable delays
        delays = []
        for ch in range(24):
            delays.append(det.get_cable_delay(station.get_id(), ch))

        # the padding is being done in the channelStopFilter module, therefor we don't need to pad the trace here
        pad_amount = int(
            max(delays)
            * sampling_rate
        )

        
        shifted_time_array = []
        shifted_volt_array = []
        
        for ch in config['channels']:

            channel_copy = copy.copy(station.get_channel(ch))
            waveform = channel_copy.get_trace()
            padded_wf = np.pad(waveform, pad_amount)

            element_shift = int(
                delays[ch] * sampling_rate 
            )
            shifted_waveform = np.roll(padded_wf, -element_shift)

            scaled_waveform = shifted_waveform / np.max(shifted_waveform) # normalize the waveform to [-1,1]
            scaled_waveform = scaled_waveform / np.std(scaled_waveform)
            scaled_waveform = scaled_waveform - np.mean(scaled_waveform)

            times = station.get_channel(ch).get_times()
            # this doesn't seem to do anyting. The `times` array from above is already the correct time array

            dt = times[1] - times[0]

            # as the traces are already padded in a previous module, we don't need to pad them here:
            # shifted time array only calculated once since time axis is the same
            shifted_times = np.array(
                [
                    float(n) * dt
                    for n in range(1, 2 * pad_amount + len(times) + 1)
                ]
            )

            shifted_volt_array.append(scaled_waveform)
            shifted_time_array.append(shifted_times)


        v_array_pairs = list(itertools.combinations(shifted_volt_array, 2))

        corr_matrix, max_corr = correlator(
            shifted_time_array, v_array_pairs, delay_matrices
        )
        if corr_map == True:
            self.plot_corr_map(corr_matrix, positions, evt=evt, config = config)

        rec_coord0, rec_coord1 = positions.get_rec_locs_from_corr_map(
            corr_matrix
        )


        if config["coord_system"] == "cylindrical":

            num_rows_to_10m = int(np.ceil(10 / abs(config['step_sizes'][1])))
            surface_corr= self.get_surf_corr(corr_matrix, num_rows_to_10m)
        else:
            num_rows_to_10m = 10
            surface_corr= self.get_surf_corr(corr_matrix, num_rows_to_10m)

            
        station.set_parameter(stnp.rec_max_correlation, max_corr)
        station.set_parameter(stnp.rec_surf_corr, surface_corr)

        if config["coord_system"] == "cylindrical":
            station.set_parameter(stnp.rec_z, rec_coord1)
            station.set_parameter(stnp.rec_azimuth, rec_coord0)
        if config["coord_system"] == "spherical":
            station.set_parameter(stnp.rec_zenith, rec_coord1)
            station.set_parameter(stnp.rec_azimuth, rec_coord0)


    
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
        rec_pulser_loc=None,
        show_actual_pulser=True,
        show_rec_pulser=True):

        lims = config['limits']

        run_number = evt.get_run_number()
        event_number = evt.get_id()
        station = evt.get_station()
        station_id = station.get_id()


        max_val = np.max(corr_matrix)
        max_idx = np.unravel_index(
            np.argmax(corr_matrix, axis=None), corr_matrix.shape
        )

        # Convert matrix indices to map coordinates
        x_extent_min, x_extent_max, y_extent_min, y_extent_max = lims
        num_rows, num_cols = corr_matrix.shape

        # Calculate the width and height of each cell in the map
        x_cell_width = (x_extent_max - x_extent_min) / num_cols
        y_cell_height = (y_extent_max - y_extent_min) / num_rows

        # Convert matrix indices to map coordinates
        # Ensure the coordinates are centered on the grid cells
        # x_coord = x_extent_min + (max_idx[1] + 0.5) * x_cell_width
        # y_coord = y_extent_min + ((num_rows - max_idx[0] - 0.5) * y_cell_height)

        mycmap = plt.get_cmap("RdBu_r")

        plt.figure(figsize=(12, 8))
        
        fig, ax = plt.subplots()

        x = np.linspace(lims[0], lims[1], corr_matrix.shape[1] + 1)
        
        #print(f"rec_coord_0 from map: {x_coord}")
        if positions.coord_system == "spherical":
            #print(f"rec_coord_1 from map: {lims[3] - y_coord}")
            #y = np.linspace(lims[2], lims[3], corr_matrix.shape[0] + 1)
            y = np.linspace(lims[2], lims[3], corr_matrix.shape[0] + 1)
            x_edges = np.linspace(x[0] - (x[1] - x[0]) / 2,
                                x[-1] + (x[1] - x[0]) / 2, corr_matrix.shape[1] + 1)
            y_edges = np.linspace(y[0] - (y[1] - y[0]) / 2,
                                y[-1] + (y[1] - y[0]) / 2, corr_matrix.shape[0] + 1)
            c = ax.pcolormesh(
                x_edges,
                y_edges,
                corr_matrix,
                cmap=mycmap,
                vmin=-np.max(corr_matrix),
                vmax=np.max(corr_matrix),
                rasterized=True,
            )
            #plt.gca().invert_yaxis()
            #plt.colorbar()
        else:
            #print(f"rec_coord_1 from map (wrt surface): {y_coord}")
            y = np.linspace(lims[2], lims[3], corr_matrix.shape[0] + 1)
            x_edges = np.linspace(x[0] - (x[1] - x[0]) / 2,
                                x[-1] + (x[1] - x[0]) / 2, corr_matrix.shape[1] + 1)
            y_edges = np.linspace(y[0] - (y[1] - y[0]) / 2,
                                y[-1] + (y[1] - y[0]) / 2, corr_matrix.shape[0] + 1)
            c = ax.pcolormesh(
                x_edges,
                y_edges,
                corr_matrix,
                cmap=mycmap,
                #edgecolors="black",
                vmin=-np.max(corr_matrix),
                vmax=np.max(corr_matrix),
                rasterized=True,
            )
            #plt.gca().invert_yaxis()
        
        x_midpoints = (x_edges[:-1] + x_edges[1:]) / 2
        y_midpoints = (y_edges[:-1] + y_edges[1:]) / 2
                
        x_midpoints = (x_edges[:-1] + x_edges[1:]) / 2
        y_midpoints = (y_edges[:-1] + y_edges[1:]) / 2
        
        max_corr_value = np.max(corr_matrix)
        max_corr_indices = np.unravel_index(np.argmax(corr_matrix), corr_matrix.shape)
        max_corr_x = x_midpoints[max_corr_indices[1]]
        max_corr_y = y_midpoints[max_corr_indices[0]]

        ax.plot(max_corr_x, max_corr_y, 'o', markersize=10, color='lime', label=f'Max corr: {max_corr_value:.2f}')
       
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        #cbar = plt.colorbar(fraction=0.046, pad=0.04)
        fig.colorbar(c)
        

        if positions.coord_system == "cylindrical":
            if positions.rec_type == "phiz":
                plt.xlabel("Azimuth Angle, $\\phi$ [$^\\circ$]", fontsize=20)
                plt.ylabel("Depth, z [m]", fontsize=20)
            elif positions.rec_type == "rhoz":
                plt.xlabel("Distance, $\\rho$ [m]", fontsize=20)
                plt.ylabel("Depth, z [m]", fontsize=20)

            # if show_rec_pulser:
            #     plt.scatter(
            #         x_coord, y_coord, color="green", label="Reconstruction"
            #     )

        else:
            plt.xlabel("Azimuth Angle, $\\phi$[$^\\circ$]", fontsize=20)
            plt.ylabel("Zenith Angle, $\\theta$[$^\\circ$]", fontsize=20)


        #plt.legend(fontsize=16)

        if positions.coord_system == "spherical": 
            plt.title(
                (
                    f"Station: {station_id}, run(s): {run_number}, "
                    + f"event: {event_number}, " 
                    + f"ch's: {config['channels']}\n"
                    + f"max_corr: {round(np.max(corr_matrix), 2)}, "
                    + f"r $\\equiv${config['fixed_coord']}m, "
                    + f"rec. loc ($\\phi$, $\\theta$): ({int(max_corr_x)}$^\\circ$, {int(lims[3] - max_corr_y)}$^\\circ$)"
                ),
                fontsize=16,
            )
        else:



            if positions.rec_type == 'phiz':
                plt.title(
                    (
                        f"Station: {station_id}, run(s): {run_number}, "
                        + f"event: {event_number}, " 
                        + f"ch's: {config['channels']}\n"
                        + f"max_corr: {round(np.max(corr_matrix), 2)}, "
                        + f"$\\rho\\equiv${config['fixed_coord']}m, rec. loc ($\\phi$, z): "
                        + f"({int(max_corr_x)}$^\\circ$, {int(max_corr_y)}m)"
                    ),
                    fontsize=16,
                )
            else:
                plt.title(
                    (
                        f"Station: {station_id}, run(s): {run_number}, "
                        + f"event: {event_number}, " 
                        + f"ch's: {config['channels']}\n"
                        + f"max_corr: {round(np.max(corr_matrix), 2)}, "
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
            print(f"Saved figure to {save_dir + f'station{station_id}_run{run_number}evt{event_number}.png'}")



class StationInfo:

    def __init__(self, station, config):

        self.station = station
        self.det = Detector(
            # json_filename=f"RNO_G/RNO_season_{year}.json"
            json_filename=config['detector_json']
        )
        self.det.update(datetime(2022, 10, 1))
        self.channels = range(0, 24)
        self.cable_delays = dict(
            zip(
                self.channels,
                [
                    self.det.get_cable_delay(int(self.station), int(channel))
                    for channel in self.channels
                ],
            )
        )

class Positions:

    def __init__(self, station_info, limits, step_sizes, coord_system, fixed_coord, rec_type):
        self.station_info = station_info
        self.limits = limits
        self.step_sizes = step_sizes
        self.coord_system = coord_system
        self.fixed_coord = fixed_coord
        self.rec_type = rec_type
        self.ant_locs = self.get_ant_locs()

        self.enu_origin = self.get_xyz_origin()
        #self.cal_locs_file = self.map_info.cal_locs_file
        
    def get_ant_locs_for_reco(self):
        """
        Returns antenna positions in the appropriate coordinate system
        for the current reconstruction type.
        """
        if self.coord_system == "cylindrical":
            return self.get_ant_locs_cylindrical(include_phi=True)
        elif self.coord_system == "spherical":
            return self.get_ant_locs_spherical()
        else:
            raise ValueError(f"Unsupported coordinate system: {self.coord_system}")

        
    def get_ant_locs_spherical(self):
        """
        Return antenna positions in spherical coordinates (r, theta, phi)
        relative to the ENU origin.

        Returns:
            dict: ch -> np.array([r, theta, phi])
        """
        spherical_locs = {}
        for ch, enu in self.ant_locs.items():
            rel = enu - self.enu_origin
            x, y, z = rel

            r = np.sqrt(x**2 + y**2 + z**2)
            theta = np.arccos(z / r) if r != 0 else 0.0  # polar angle from z-axis
            phi = np.arctan2(y, x)
            if phi < 0:
                phi += 2 * np.pi  # wrap to [0, 2π]

            spherical_locs[ch] = np.array([r, theta, phi])
        return spherical_locs


    def get_ant_locs_cylindrical(self, include_phi=False):
        """
        Return antenna positions in cylindrical coordinates (r, z) or (r, z, phi),
        relative to the station ENU origin.

        Args:
            include_phi (bool): whether to include phi (in radians) in the result

        Returns:
            dict: ch -> np.array([r, z]) or np.array([r, z, phi])
        """
        r_z_locs = {}
        for ch, enu in self.ant_locs.items():
            
            rel = enu - self.enu_origin

            x, y, z = rel
            
            r = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x) if include_phi else None
            phi = np.where(phi < 0, phi + 2 * np.pi, phi)
            
            
            r_z_locs[ch] = np.array([r, z]) if not include_phi else np.array([r, z, phi])
        return r_z_locs

    def get_xyz_origin(self):

        # if self.cal_locs_file:
        #     xyz_ch_1_loc = np.array(self.get_ant_locs()[0]['1'])
        #     xyz_ch_2_loc = np.array(self.get_ant_locs()[0]['2'])
        # else:
        # xyz_ch_1_loc = self.station_info.det.get_relative_position(
        #     int(self.station_info.station), 1
        # )
        xyz_ch_1_loc = self.ant_locs[1]
        xyz_ch_2_loc = self.ant_locs[2]
        xyz_origin_loc = (xyz_ch_1_loc + xyz_ch_2_loc) / 2.0

        return xyz_origin_loc

    def get_ant_locs(self):

        all_ch_ids = range(0, 24)
        # if self.cal_locs_file:
        #     data = np.load(self.cal_locs_file)

        #     all_xyz_ant_locs = {}
        #     for channel_idx, channel in enumerate(data):
        #         all_xyz_ant_locs[str(channel_ids[channel_idx])] = [
        #             channel[0][231],
        #             channel[1][231],
        #             channel[2][231],
        #         ]
        # else:
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
        From specified limits and step size, create arrays for each pulser
            coordinate in its proper units.
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
                #coord0_vec = np.array(coord0_vec) * units.deg
                coord1_vec = [coord1 * units.m for coord1 in coord1_vec]
            elif self.rec_type == "rhoz":
                coord0_vec = [coord0 * units.m for coord0 in coord0_vec]
                coord1_vec = [coord1 * units.m for coord1 in coord1_vec]
        elif self.coord_system == "spherical":
            coord0_vec = [coord0 * units.deg for coord0 in coord0_vec]
            coord1_vec = [coord1 * units.deg for coord1 in coord1_vec]

        return coord0_vec, coord1_vec

    def get_coord_grids(self):
        coord0_vec, coord1_vec = self.generate_coord_arrays()

        if self.coord_system == "cylindrical":
            if self.rec_type == "phiz":
                coord2_grid, coord1_grid = np.meshgrid(coord0_vec, coord1_vec)  # phi, z
                coord0_grid = np.full_like(coord2_grid, self.fixed_coord)       # rho
            elif self.rec_type == "rhoz":
                coord0_grid, coord1_grid = np.meshgrid(coord0_vec, coord1_vec)  # rho, z
                coord2_grid = np.full_like(coord0_grid, self.fixed_coord)       # phi
            else:
                raise ValueError(f"Invalid rec_type: {self.rec_type}")

        elif self.coord_system == "spherical":
            coord2_grid, coord1_grid = np.meshgrid(coord0_vec, coord1_vec)      # phi, theta
            coord0_grid = np.full_like(coord2_grid, self.fixed_coord)           # r

        else:
            raise ValueError(f"Unsupported coordinate system: {self.coord_system}")

        return coord0_grid, coord1_grid, coord2_grid


    def get_source_enu_matrix(self, rec_type):
        """
        Returns a matrix of potential source locations in enu coords with
            respect to the station origin. It must be in this
            coord system for the time difference calculator to work properly.
        """
        coord0_vec, coord1_vec = self.generate_coord_arrays()

        if self.coord_system == "cylindrical":
            if rec_type == "phiz":
                rho_to_pulser = self.fixed_coord

                phi_grid, z_grid = np.meshgrid(coord0_vec, coord1_vec)
                rho_grid = np.full_like(phi_grid, rho_to_pulser)

                x_grid, y_grid, z_grid = self.get_enu_wrt_new_origin(
                    [rho_grid, phi_grid, z_grid]
                )

            elif rec_type == "rhoz":
                phi = self.fixed_coord

                rho_grid, z_grid = np.meshgrid(coord0_vec, coord1_vec)

                phi_grid = np.full_like(rho_grid, phi)

                x_grid, y_grid, z_grid = self.get_enu_wrt_new_origin(
                    [rho_grid, phi_grid, z_grid]
                )

        else:
            rval = self.fixed_coord
            phi_grid, theta_grid = np.meshgrid(coord0_vec, coord1_vec)
            rho_grid = np.full_like(phi_grid, rval)

            x_grid, y_grid, z_grid = self.get_enu_wrt_new_origin(
                [rho_grid, phi_grid, theta_grid]
            )

        src_xyz_loc_matrix = np.stack((x_grid, y_grid, z_grid), axis=-1)

        x_values = src_xyz_loc_matrix[:, :, 0]
        y_values = src_xyz_loc_matrix[:, :, 1]
        z_values = src_xyz_loc_matrix[:, :, 2]

        # Compute min and max for x, y, z
        min_x, max_x = np.min(x_values), np.max(x_values)
        min_y, max_y = np.min(y_values), np.max(y_values)
        min_z, max_z = np.min(z_values), np.max(z_values)

        # print(f"Min X: {min_x:.2f}, Max X: {max_x:.2f}")
        # print(f"Min Y: {min_y:.2f}, Max Y: {max_y:.2f}")
        # print(f"Min Z: {min_z:.2f}, Max Z: {max_z:.2f}")

        return src_xyz_loc_matrix, [coord0_vec, coord1_vec], (rho_grid, z_grid, phi_grid)


    def get_enu_wrt_new_origin(self, coords):
        """
        Converts enu to enu of xyz_origin coord system
        """
        enu_origin = self.get_xyz_origin()
        #print(f"enu origin: {enu_origin}", flush=True)

        if self.coord_system == "cylindrical":
            rhos = coords[0]
            phis = coords[1]
            zs = coords[2]

            eastings = rhos * np.cos(phis)
            northings = rhos * np.sin(phis)
            elevations = zs

        else:
            rs = coords[0]
            phis = coords[1]
            thetas = coords[2]

            eastings = rs * np.sin(thetas) * np.cos(phis)
            northings = rs * np.sin(thetas) * np.sin(phis)
            elevations = rs * np.cos(thetas)

        if self.coord_system == "spherical":
            eastings = eastings + enu_origin[0]
            northings = northings + enu_origin[1]
            elevations = elevations + enu_origin[2]

        return eastings, northings, elevations

    def get_rec_locs_from_corr_map(self, corr_matrix):

        coord0_vec, coord1_vec = self.generate_coord_arrays()
        
        rec_pulser_loc0_idx, rec_pulser_loc1_idx = get_max_val_indices(
            corr_matrix
        )
        coord0_best = coord0_vec[rec_pulser_loc0_idx]
        coord1_best = coord1_vec[rec_pulser_loc1_idx]

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

    from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
    from NuRadioReco.detector import detector
    import argparse
    import os
    from datetime import datetime
    #import datetime

    parser = argparse.ArgumentParser(prog="%(prog)s", usage="reconstruction test")
    parser.add_argument("--station", type=int, default=21)
    parser.add_argument("--event", type = int, default = 4241)
    parser.add_argument("--run", type=int, default=240430)
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")

    args = parser.parse_args()

    data_dir = os.environ["RNO_G_DATA"] # used deep CR burn sample..

    readRNOGData = readRNOGData()  

    root_dirs = f"{data_dir}/station{args.station}/run{args.run}/combined.root"
    readRNOGData.begin(root_dirs)
    print(f"Event {args.event}, Station {args.station}, Run {args.run}")

    reco = directionReconstructionDeepCRsearch()
    reco.begin()

    #logger.setLevel(logging.DEBUG)
    event=readRNOGData.get_event(args.run, args.event)
    station=event.get_station()

    det = detector.Detector(source="rnog_mongo")
    det.update(datetime(2022, 10, 1))

    reco.run(event, station, det, args.config, corr_map=True)
    print(f"Max correlation: {station.get_parameter(stnp.rec_max_correlation)}")
    print(f"Surface correlation: {station.get_parameter(stnp.rec_surf_corr)}")
    print(f"Reconstructed azimuth: {station.get_parameter(stnp.rec_azimuth)} deg")
    print(f"Reconstructed z: {station.get_parameter(stnp.rec_z)/units.m} deg")
    reco.end()
    readRNOGData.end()
