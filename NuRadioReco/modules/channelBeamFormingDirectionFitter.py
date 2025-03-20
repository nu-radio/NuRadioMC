import matplotlib.pyplot as plt
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units
import NuRadioReco.detector.detector
import uproot
import numpy as np
import itertools
import argparse
#from plotting import plotter
import datetime
import sys, os
from scipy import signal
import yaml
from scipy.interpolate import RegularGridInterpolator
from NuRadioReco.modules.RNO_G.channelBlockOffsetFitter import fit_block_offsets
import pandas as pd
import copy
import json
from NuRadioReco.modules.base import module
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.utilities.lookup_table_creator import compute_time_differences
# Now import works
#from tools.lookup_table_creator import compute_time_differences  # Import function

def get_interpolator(station, ch_pair):
    """Check if the time delay lookup table exists; if not, generate it and load the interpolator."""
    #filename = f"/mnt/ceph1-npx/user/anozdrina/rno-g/cr_analysis/interp_tables/station{station}/station{station}_2023json_time_differences_3d_{ch_pair[0]}_{ch_pair[1]}_200_1deg_grid_noraytracing.npy"
    add = "/data/user/anozdrina/rno-g/lookup_tables/interp_tables/"
    filename = f"station{station}/station{station}_2023json_time_differences_3d_{ch_pair[0]}_{ch_pair[1]}_200_1deg_grid_noraytracing.npy"
    filename = add+filename
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Lookup table {filename} not found. Cannot continue execution.")#to run in condor
        # print(f"Lookup table {filename} not found. Generating it now...")
        # compute_time_differences(station, ch_pair)

    return load_interpolator(filename)

class channelBeamFormingDirectionFitter():
    def __init__(self):
        self.begin()

    def begin(self):
        pass

    @register_run()
    def run(self, evt, station, det, config):
        
        if isinstance(config, str):
            config = self.load_config(config)

        map_info = MapInfo(
            config["limits"],
            config["stepsize"],
            config["channels"],
            config["calibrate"],
            config["events"],
        )
        station_info = StationInfo(config["station"], det)

        channels = map_info.used_channels
        positions = Positions(
            station_info,
            map_info,
            config["coords"],
            config["distance"],
            config["rec_type"],
        )

        ant_locs = positions.get_ant_locs()[1]

        # load time delay tables, generated for each pair of channels and for a grid of source locations
        delay_matrices = positions.get_t_delay_matrices(
            ant_locs, config["rec_type"]
        )
        #initialize variables to save
        corr_matrix_sum = None
        max_corr_vals = -1.0
        # rec_coord0s = []
        # rec_coord1s = []
        surface_corr = -1.0

#already applied time delays to the waveforms in the channelBlockOffsetFitter module
                # the current sampling rate can be requested from the station object
        sampling_rate = station.get_channel(0).get_sampling_rate()  #assuming that there is a channel 0

        delays = []
        for ch in range(24):
            delays.append(det.get_cable_delay(station.get_id(), ch))
        #     print(ch)
        # print(delays)
        # the padding is being done in the channelStopFilter module, therefor we don't need to pad the trace here
        pad_amount = int(
            max(delays)
            * sampling_rate
        )

        times = station.get_channel(0).get_times()

        # this doesn't seem to do anyting. The `times` array from above is already the correct time array
        new_times = np.linspace(
            times[0], times[-1], len(times)
        )
        dt = times[1] - times[0]

        # as the traces are already padded in a previous module, we don't need to pad them here:
        # shifted time array only calculated once since time axis is the same
        #   for all waveforms in a run
        shifted_time_array = np.array(
            [
                float(n) * dt
                for n in range(1, 2 * pad_amount + len(times) + 1)
            ]
        )

        shifted_volt_array = []
        
        for ch in map_info.used_channels:
            # channel = station.get_channel(ch)
            # shifted_waveform = channel.get_trace()
            channel_copy = copy.copy(station.get_channel(ch))
            waveform = channel_copy.get_trace()
            padded_wf = np.pad(waveform, pad_amount)
            # print(ch, delays[ch])
            element_shift = int(
                delays[ch] * sampling_rate 
            )
            shifted_waveform = np.roll(padded_wf, -element_shift)

            # channel_copy = copy.copy(station.get_channel(ch))
            # channel_copy.apply_time_shift(det.get_cable_delay(station.get_id(), channel_copy.get_id()))
            # shifted_waveform = channel_copy.get_trace()

            scaled_waveform = shifted_waveform / np.max(shifted_waveform) # normalize the waveform to [-1,1]
            scaled_waveform = scaled_waveform / np.std(scaled_waveform)
            scaled_waveform = scaled_waveform - np.mean(scaled_waveform)

            shifted_volt_array.append(scaled_waveform)


        v_array_pairs = list(itertools.combinations(shifted_volt_array, 2))

        corr_matrix, max_corr = self.correlator(
            shifted_time_array, v_array_pairs, delay_matrices
        )

        #self.plot_corr_map(corr_matrix, positions, map_info, f"station{config['station']}_run{config['run']}_testing.png",event=evt)

        rec_coord0, rec_coord1 = positions.get_rec_locs_from_corr_map(
            corr_matrix
        )


        if config["coords"] == "cylindrical":
            num_rows_to_10m = int(config["limits"][-1] / 10) + 1
            surface_corr= self.get_surf_corr_ratio(corr_matrix, num_rows_to_10m)
        else:
            num_rows_to_10m = 10
            surface_corr= self.get_surf_corr_ratio(corr_matrix, num_rows_to_10m)

            
        station.set_parameter(stnp.rec_max_correlation, max_corr)
        station.set_parameter(stnp.rec_surf_corr, surface_corr)
        station.set_parameter(stnp.rec_zenith, rec_coord1)
        station.set_parameter(stnp.rec_azimuth, rec_coord0)
        # to safe output parameters, add new parameters to station or channel level in `NuRadioReco/framework/parameters.py`
        # then set the parameter like this:
        # e.g. station.set_parameter(stnp.beamforming_direction, rec_coord0)
        # the eventWriter module will automatically write them to disk. 
        # you can also use the `channelParameters` class to set parameters on the channel level.
        # if you want to access the parameters in a later module, you can use the `get_parameter` method of the station or channel object.
        # e.g. rec_coord0 = station.get_parameter(stnp.beamforming_direction)
        
    
    
    def end(self):
        pass

    def load_config(self, config_file):
        with open(config_file, "r") as f:
            return yaml.safe_load(f)

    def get_surf_corr_ratio(self, corr_map, num_rows_for_10m):
        surf_corr_ratio = np.max(corr_map[:num_rows_for_10m]) / np.max(corr_map)
        return surf_corr_ratio

    def load_interpolator(self, table_filename):
        time_differences_fine_3d = np.load(table_filename)

        x_range = np.arange(-200, 201, 1)
        y_range = np.arange(-200, 201, 1)
        z_range = np.arange(-100, 1, 1)

        interpolator = RegularGridInterpolator(
            (x_range, y_range, z_range), time_differences_fine_3d, method="linear"
        )

        return interpolator

    def corr_index_from_t(self, time_delay, times):
        # It gives an array of integer indices that correspond to the time_delay values mapped onto the times array.
        # These indices can then be used for lookups or correlation calculations
        center = len(times)
        dt = times[1] - times[0]

        time_delay_clean = np.nan_to_num(time_delay, nan=0.0)

        index = np.rint(time_delay_clean / dt + (center - 1)).astype(int)

        return index

    def get_max_val_indices(self, matrix):
        max_value = matrix.argmax()
        row_by_col = matrix.shape
        best_row_index, best_col_index = np.unravel_index(max_value, row_by_col)
        return best_col_index, best_row_index

    def correlator(self, times, v_array_pairs, delay_matrices):

        amplitude_corrector = 1.0 / (float(len(v_array_pairs[0][0]))) # scale by the first waveform ?? 

        volt_corrs = []
        for v1, v2 in v_array_pairs:
            correlation = signal.correlate(v1, v2)  # Compute cross-correlation
            scaled_correlation = amplitude_corrector * np.asarray(correlation)  # Scale it
            volt_corrs.append(scaled_correlation)  # Store the result

        pair_corr_matrices = []

        for volt_corr, time_delay in zip(volt_corrs, delay_matrices):
            indices = self.corr_index_from_t(time_delay, times)
            mask = np.isnan(time_delay)
            corr_matrix = np.zeros_like(time_delay)

            for i in range(indices.shape[0]):
                for j in range(indices.shape[1]):
                    if not mask[i, j]:
                        corr_matrix[i, j] = volt_corr[indices[i, j]]

            pair_corr_matrices.append(corr_matrix)

        mean_corr_matrix = np.mean(np.array(pair_corr_matrices), axis=0)
        max_corr = np.max(np.array(mean_corr_matrix))

        return mean_corr_matrix, max_corr

    def plot_corr_map(
        self,
        corr_matrix, positions, map_info,
        file_name=None,
        event=None,
        rec_pulser_loc=None,
        show_actual_pulser=True,
        show_rec_pulser=True):

        lims = [0, 360, 90, 0]
        lims = [lims[0], lims[1], lims[2], lims[3]]
        
        #lims = [0, 130, -94.7545, -4.7545]

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

        # #Add text for each value at the calculated midpoints
        # for i in range(corr_matrix.shape[0]):
        #     for j in range(corr_matrix.shape[1]):
        #         ax.text(x_midpoints[j], y_midpoints[i], f"{corr_matrix[i, j]:.2f}", 
        #                 ha="center", va="center", color="black", fontsize=1)
                
        x_midpoints = (x_edges[:-1] + x_edges[1:]) / 2
        y_midpoints = (y_edges[:-1] + y_edges[1:]) / 2
        
        max_corr_value = np.max(corr_matrix)
        max_corr_indices = np.unravel_index(np.argmax(corr_matrix), corr_matrix.shape)
        max_corr_x = x_midpoints[max_corr_indices[1]]
        max_corr_y = y_midpoints[max_corr_indices[0]]
        # print(f"\nmax corr x: {max_corr_x}")
        # print(f"\nmax corr y: {max_corr_y}")
        ax.plot(max_corr_x, max_corr_y, 'o', markersize=10, color='lime', label=f'Max corr: {max_corr_value:.2f}')
        #plt.savefig("/data/user/anozdrina/rno-g/cr_analysis/corr_map.png", dpi=300, bbox_inches='tight')  # Save the figure
                
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        #cbar = plt.colorbar(fraction=0.046, pad=0.04)
        fig.colorbar(c)
        
        # for t in cbar.ax.get_yticklabels():
        #     t.set_fontsize(20)

        # if self.positions.coord_system == "spherical":
        #     plt.scatter(x_coord, lims[3]-y_coord, label='Rec. point', s=30)
        # else:
        #     plt.scatter(x_coord, y_coord, label='Rec. point', s=30)
  

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

        if map_info.cal_locs_file is not None:
            cal_status = "cal"
        else:
            cal_status = "uncal"

        #plt.legend(fontsize=16)

        if positions.coord_system == "spherical": 
            plt.title(
                (
                    # f"Station: {self.station}, run(s) {self.run}, "
                    # + (f"event: {event}, " if event is not None else "all events, ")
                    # + f"ch's: {map_info.used_channels}, \n"
                    # + f"max_corr: {round(np.max(corr_matrix), 2)}, "
                    # + f"r $\\equiv${self.distance}m, "
                    # + f"rec. loc ($\\phi$, $\\theta$): ({int(max_corr_x)}$^\\circ$, {int(lims[3] - max_corr_y)}$^\\circ$)"
                ),
                fontsize=16,
            )
        else:
            if positions.rec_type == 'phiz':
                plt.title(
                    (
                        # f"Station: {self.station}, run(s): {self.run}, "
                        # + (f"event: {event}, " if event is not None else "all events, ")
                        # + f"ch's: {map_info.used_channels}\n"
                        # + f"max_corr: {round(np.max(corr_matrix), 2)}, "
                        # + f"$\\rho\\equiv${self.distance}m, rec. loc ($\\phi$, z): "
                        # + f"({int(max_corr_x)}$^\\circ$, {int(max_corr_y)}m)"
                    ),
                    fontsize=16,
                )
            else:
                plt.title(
                    (
                        # f"Station: {self.station}, run(s): {self.run}, "
                        # + (f"event: {event}, " if event is not None else "all events, ")
                        # + f"ch's: {self.chs}\n"
                        # + f"max_corr: {round(np.max(corr_matrix), 2)}, "
                        # + f"rec. loc ($\\rho$, z): ({int(max_corr_x)}m, {int(max_corr_y)}m)"
                    ),
                    fontsize=16,
                )
        save_dir = "/data/user/anozdrina/rno-g/cr_analysis/"
        plt.tight_layout()
        if show_actual_pulser or show_rec_pulser:
            plt.legend()
        if file_name is not None:
            plt.savefig(save_dir + file_name)
        else:
            plt.savefig(
                save_dir + "corr_map_event__testcluster.png"
            )
            print(f"Saved figure to {save_dir}corr_map_event__testcluster.png")

def get_surf_corr_ratio(corr_map, num_rows_for_10m):
    # print(f"slice of corr: {corr_map[:num_rows_for_10m]}")
    surf_corr_ratio = np.max(corr_map[:num_rows_for_10m]) / np.max(corr_map)

    return surf_corr_ratio


def load_interpolator(table_filename):
    time_differences_fine_3d = np.load(table_filename)

    # Define the grid used for generating lookup tables
    x_range = np.arange(-200, 201, 1)
    y_range = np.arange(-200, 201, 1)
    z_range = np.arange(-100, 1, 1)

    # Create the interpolator
    interpolator = RegularGridInterpolator(
        (x_range, y_range, z_range), time_differences_fine_3d, method="linear"
    )

    return interpolator


class MapInfo:

    def __init__(
        self,
        limits,
        stepsize,
        used_channels,
        cal_locs_file,
        selected_events,
    ):
        self.limits = limits
        self.stepsize = stepsize
        self.used_channels = used_channels
        self.cal_locs_file = cal_locs_file
        self.selected_events = selected_events
        

class StationInfo:

    def __init__(self, station, det):
        self.det = det
        self.station = station
        det.update(datetime.datetime(2022, 10, 1))
        self.channels = range(0, 24)
        self.delays = dict(
            zip(
                self.channels,
                [
                    self.det.get_cable_delay(int(self.station), int(channel))
                    for channel in self.channels
                ],
            )
        )



class Positions:

    def __init__(self, station_info, map_info, coord_system, dist, rec_type):
        self.rec_type = rec_type
        self.phis = None
        self.map_info = map_info
        self.station_info = station_info
        self.coord_system = coord_system
        self.cal_locs_file = self.map_info.cal_locs_file
        self.pulser_id = None
        self.distance = dist



    def get_t_delay_matrices(self, ant_locs, rec_type):

        ch_pairs = list(itertools.combinations(self.map_info.used_channels, 2))
        # print(f"ch pairs used: {ch_pairs}\n")
        posn_pairs = list(itertools.combinations(ant_locs, 2))

        z_offset = -self.get_xyz_origin()[2]

        src_enu_matrix_and_coord_vecs = self.get_source_enu_matrix(rec_type)
        src_posn_enu_matrix = src_enu_matrix_and_coord_vecs[0]


        # for src_posn_enu_matrix in src_posn_enu_matrices:

        time_delay_matrices = []
        for ch_pair, posn_pair in zip(ch_pairs, np.array(posn_pairs)):
            my_interp = get_interpolator(self.station_info.station, ch_pair)
            time_delay_matrix = my_interp(src_posn_enu_matrix)
            time_delay_matrices.append(time_delay_matrix)

        return time_delay_matrices

    def get_xyz_origin(self):

        if self.cal_locs_file:
            xyz_ch_1_loc = np.array(self.get_ant_locs()[0]["1"])
            xyz_ch_2_loc = np.array(self.get_ant_locs()[0]["2"])
        else:
            xyz_ch_1_loc = self.station_info.det.get_relative_position(
                int(self.station_info.station), 1
            )
            xyz_ch_2_loc = self.station_info.det.get_relative_position(
                int(self.station_info.station), 2
            )

        xyz_origin_loc = (xyz_ch_1_loc + xyz_ch_2_loc) / 2.0

        return xyz_origin_loc

    def get_ant_locs(self):

        channel_ids = [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23]
        if self.cal_locs_file:
            data = np.load(self.cal_locs_file)

            all_xyz_ant_locs = {}
            for channel_idx, channel in enumerate(data):
                all_xyz_ant_locs[str(channel_ids[channel_idx])] = [
                    channel[0][231],
                    channel[1][231],
                    channel[2][231],
                ]
        else:
            all_xyz_ant_locs = {}
            for ch in channel_ids:
                all_xyz_ant_locs[str(ch)] = (
                    self.station_info.det.get_relative_position(
                        int(self.station_info.station), int(ch)
                    ).tolist()
                )

        used_xyz_ant_locs = [
            all_xyz_ant_locs[str(ch)] for ch in self.map_info.used_channels
        ]

        return all_xyz_ant_locs, used_xyz_ant_locs

    def generate_coord_arrays(self):
        """
        From specified limits and step size, create arrays for each pulser
            coordinate in its proper units.
        """
        left, right, bottom, top = self.map_info.limits

        coord0_vec = np.arange(
            left, right + self.map_info.stepsize, self.map_info.stepsize
        )
        coord1_vec = np.arange(
            bottom, top - self.map_info.stepsize, -self.map_info.stepsize
        )

        # else:
        #     coord1_vec = np.arange(bottom, top + self.map_info.stepsize, self.map_info.stepsize)

        # print(f"coord0 vec: {coord0_vec}")
        # print(f"coord1 vec: {coord1_vec}")

        if coord0_vec[-1] < right:
            coord0_vec = np.append(coord0_vec, right)
        if coord1_vec[-1] < top:
            coord1_vec = np.append(coord1_vec, top)

        if self.rec_type == "surface":
            coord0_vec = [coord0 * units.m for coord0 in coord0_vec]
            coord1_vec = [coord1 * units.deg for coord1 in coord1_vec]
        else:
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

                # print(f"coord1_vec: {[angle/units.deg for angle in coord1_vec]}")

            else:
                sys.exit(
                    "Coordinate system not found. Please specify either "
                    "'cylindrical' or 'spherical' coordinates."
                )


        return coord0_vec, coord1_vec

    def get_surface_coords(self, coord0_vec, coord1_vec):

        surface_coord_dict = defaultdict(list)
        x = []
        y = []
        z = [-self.get_xyz_origin()[2]] * (len(coord0_vec) * len(coord1_vec))
        for R in coord0_vec:
            for phi in coord1_vec:
                x.append(R * np.cos(phi / units.rad))
                y.append(R * np.sin(phi / units.rad))

                surface_coord_dict[(R, phi)] = [x, y, z]

        return surface_coord_dict

    def get_source_enu_matrix(self, rec_type):
        """
        Returns a matrix of potential source locations in enu coords with
            respect to the station origin. It must be in this
            coord system for the time difference calculator to work properly.
        """
        coord0_vec, coord1_vec = self.generate_coord_arrays()

        if self.coord_system == "cylindrical":
            if rec_type == "phiz":
                rho_to_pulser = self.distance

                phi_grid, z_grid = np.meshgrid(coord0_vec, coord1_vec)
                rho_grid = np.full_like(phi_grid, rho_to_pulser)

                x_grid, y_grid, z_grid = self.get_enu_wrt_new_origin(
                    [rho_grid, phi_grid, z_grid]
                )

            elif rec_type == "rhoz":
                phi = 0 * units.deg

                rho_grid, z_grid = np.meshgrid(coord0_vec, coord1_vec)

                # rho_grid = rho_grid[:len(coord1_vec)-1, :len(coord0_vec)-1]
                # z_grid = z_grid[:len(coord1_vec)-1, :len(coord0_vec)-1]

                phi_grid = np.full_like(rho_grid, phi)

                x_grid, y_grid, z_grid = self.get_enu_wrt_new_origin(
                    [rho_grid, phi_grid, z_grid]
                )

        else:
            rval = self.distance
            phi_grid, theta_grid = np.meshgrid(coord0_vec, coord1_vec)
            r_grid = np.full_like(phi_grid, rval)

            x_grid, y_grid, z_grid = self.get_enu_wrt_new_origin(
                [r_grid, phi_grid, theta_grid]
            )

        src_xyz_loc_matrix = np.stack((x_grid, y_grid, z_grid), axis=-1)

        x_values = src_xyz_loc_matrix[:, :, 0]
        y_values = src_xyz_loc_matrix[:, :, 1]
        z_values = src_xyz_loc_matrix[:, :, 2]

        # Compute min and max for x, y, z
        min_x, max_x = np.min(x_values), np.max(x_values)
        min_y, max_y = np.min(y_values), np.max(y_values)
        min_z, max_z = np.min(z_values), np.max(z_values)

        # Print results rounded to 2 decimal places
        # print(f"Min X: {min_x:.2f}, Max X: {max_x:.2f}")
        # print(f"Min Y: {min_y:.2f}, Max Y: {max_y:.2f}")
        # print(f"Min Z: {min_z:.2f}, Max Z: {max_z:.2f}")

        # print("\n")
        # print(src_xyz_loc_matrix)
        # print(np.shape(src_xyz_loc_matrix))
        # print("\n")

        # print(src_xyz_loc_matrix[0][0])

        # print(f"\nsrc loc matrix: {src_xyz_loc_matrix}\n")

        # print(f"\nsize of src matrix: {np.shape(src_xyz_loc_matrix)}")

        return src_xyz_loc_matrix, [coord0_vec, coord1_vec]

    def get_enu_wrt_new_origin(self, coords):
        """
        Converts enu to enu of xyz_origin coord system
        """
        enu_origin = self.get_xyz_origin()

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

        # prob should re-enable this?

        if self.coord_system == "spherical":
            # prob should re-enable for the rest too?
            eastings = eastings + enu_origin[0]
            northings = northings + enu_origin[1]
            elevations = elevations + enu_origin[2]

        # print(f"\n\nenu origin: {enu_origin}\n\n")
        # print(f"\n\nenu origin z: {enu_origin[2]}\n\n")

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

    max_value = matrix.argmax()
    row_by_col = matrix.shape
    best_row_index, best_col_index = np.unravel_index(max_value, row_by_col)

    return best_col_index, best_row_index


