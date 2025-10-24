"""
Correlation Map Plotter

This script reads saved correlation map data (pickle files) and generates publication-quality 
plots of the correlation maps with reconstruction results, alternate coordinates, exclusion zones,
and optional minimap insets.

Usage:
    python correlation_map_plotter.py --input correlation_maps/
    python correlation_map_plotter.py --input station21_run476_evt7_corrmap.pkl --output custom_plot.png
    python correlation_map_plotter.py --input correlation_maps/ --pattern "*run476*" --minimaps
"""

import argparse
import os
import glob
import re
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from NuRadioReco.utilities import units
from NuRadioReco.utilities.interferometry_io_utilities import (
    load_correlation_map,
    determine_plot_output_path
)
from NuRadioReco.modules.io.NuRadioRecoio import NuRadioRecoio
from NuRadioReco.modules.io.eventReader import eventReader
from NuRadioReco.modules.channelResampler import channelResampler
from NuRadioReco.modules.channelBandPassFilter import channelBandPassFilter
from NuRadioReco.modules.channelSinewaveSubtraction import channelSinewaveSubtraction
from NuRadioReco.modules.channelAddCableDelay import channelAddCableDelay
from NuRadioReco.detector.RNO_G import rnog_detector
from NuRadioReco.framework.parameters import particleParameters
from NuRadioReco.framework.parameters import showerParameters
from NuRadioReco.framework.parameters import generatorAttributes
from ray_path import plot_ray_paths

plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 22,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
})

class CorrelationMapPlotter:
    """
    Class for plotting correlation maps in various formats.
    
    Supports both simple correlation map plots and comprehensive multi-panel visualizations
    combining correlation maps, waveforms, and event metadata.
    """
    
    def __init__(self, map_data_path=None, output_arg=None, show_minimaps=False, extra_points=None):
        """
        Initialize the plotter with common settings.
        
        Parameters
        ----------
        map_data_path : str, optional
            Path to correlation map pickle file
        output_arg : str, optional
            Output directory or file path
        show_minimaps : bool, optional
            Whether to show minimap insets (default: False)
        extra_points : list, optional
            Extra points to plot on correlation map
        """
        self.map_data_path = map_data_path
        self.output_arg = output_arg
        self.show_minimaps = show_minimaps
        self.extra_points = extra_points if extra_points is not None else []
        
        # Load the correlation map data once
        if self.map_data_path is not None:
            self.map_data = load_correlation_map(self.map_data_path)
        else:
            self.map_data = None

    def plot_correlation_map(self, ax=None, fig=None, standalone=True, extra_points_override=None, force_minimaps=None):
        """
        Generate correlation map plot from saved data.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure (default: None)
        fig : matplotlib.figure.Figure, optional
            Figure object for colorbar placement. Required if ax is provided.
        standalone : bool, optional
            If True, saves figure and returns path. If False, just plots on ax and returns ax (default: True)
        extra_points_override : list, optional
            Extra points to plot, overriding self.extra_points if provided (default: None)
        force_minimaps : bool, optional
            If provided, overrides self.show_minimaps for this call (default: None)
        
        Returns
        -------
        str or matplotlib.axes.Axes
            Path to saved plot file if standalone=True, otherwise the axes object
        """
        
        if self.map_data is None:
            print("Error: No correlation map data loaded")
            return None
        
        map_data = self.map_data
        corr_matrix = map_data['corr_matrix']
        station_id = map_data['station_id']
        run_number = map_data['run_number']
        event_number = map_data['event_number']
        config = map_data['config']
        coord_system = map_data['coord_system']
        rec_type = map_data.get('rec_type')
        limits = map_data['limits']
        
        show_minimaps = force_minimaps if force_minimaps is not None else self.show_minimaps
        create_minimaps = show_minimaps
        
        mycmap = plt.get_cmap("RdBu_r")
        mycmap.set_bad(color='black')

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        elif fig is None:
            raise ValueError("Must provide both ax and fig together")

        c = ax.imshow(
            corr_matrix,
            cmap=mycmap,
            vmin=np.nanmin(corr_matrix),
            vmax=np.nanmax(corr_matrix),
            extent=[limits[0], limits[1], limits[2], limits[3]],
            origin='lower',
            interpolation='nearest',
            aspect='auto',
            rasterized=True,
        )

        max_corr_x = map_data.get('coord0', None)
        max_corr_y = map_data.get('coord1', None)
        max_corr_value = map_data.get('max_corr', np.nan)
        
        # If coordinates not in map_data, compute from correlation matrix
        if max_corr_x is None or max_corr_y is None:
            max_idx = np.unravel_index(np.nanargmax(corr_matrix), corr_matrix.shape)
            max_corr_x = coord0_vec[max_idx[1]]
            max_corr_y = coord1_vec[max_idx[0]]
            max_corr_value = corr_matrix[max_idx]
        
        # Format legend label based on coordinate system
        if coord_system == "cylindrical":
            if rec_type == "phiz":
                max_corr_x = np.rad2deg(max_corr_x)
                legend_label = f"Max corr: {max_corr_value:.2f} at ({max_corr_x:.2f}°, {max_corr_y:.2f}m)"
            elif rec_type == "rhoz":
                legend_label = f"Max corr: {max_corr_value:.2f} at ({max_corr_x:.2f}m, {max_corr_y:.2f}m)"
        elif coord_system == "spherical":
            max_corr_x = np.rad2deg(max_corr_x)
            max_corr_y = np.rad2deg(max_corr_y)
            legend_label = f"Max corr: {max_corr_value:.2f} at ({max_corr_x:.2f}°, {max_corr_y:.2f}°)"
        
        ax.plot(
            max_corr_x,
            max_corr_y,
            "o",
            markersize=6,
            color="lime",
            label=legend_label,
        )
        
        if 'coord0_alt' in map_data and map_data['coord0_alt'] is not None:
            coord0_alt = map_data['coord0_alt']
            coord1_alt = map_data['coord1_alt']
            
            if not np.isnan(coord0_alt) and not np.isnan(coord1_alt):
                # Build legend label with coordinates
                if coord_system == "cylindrical" and rec_type == "phiz":
                    coord0_alt = np.rad2deg(coord0_alt)
                    alt_label = f"Alt max: ({coord0_alt:.2f}°, {coord1_alt:.2f}m)"
                elif coord_system == "cylindrical" and rec_type == "rhoz":
                    alt_label = f"Alt max: ({coord0_alt:.2f}m, {coord1_alt:.2f}m)"
                elif coord_system == "spherical":
                    coord0_alt = np.rad2deg(coord0_alt)
                    coord1_alt = np.rad2deg(coord1_alt)
                    alt_label = f"Alt max: ({coord0_alt:.2f}°, {coord1_alt:.2f}°)"
                else:
                    alt_label = "Alt max"
                
                ax.plot(
                    coord0_alt,
                    coord1_alt,
                    "o",
                    markersize=6,
                    color="lime",
                    fillstyle="none",
                    markeredgewidth=1,
                    label=alt_label,
                )
        
        # Plot extra points (like true vertex) AFTER max and alt max, with proper units
        points_to_plot = extra_points_override if extra_points_override is not None else self.extra_points
        for x, y, label in points_to_plot:
            # Format coordinates with proper units based on coordinate system
            if coord_system == "cylindrical":
                if rec_type == "phiz":
                    coord_label = f"{label}({x:.2f}°, {y:.2f}m)"
                elif rec_type == "rhoz":
                    coord_label = f"{label}({x:.2f}m, {y:.2f}m)"
            elif coord_system == "spherical":
                coord_label = f"{label}({x:.2f}°, {y:.2f}°)"
            else:
                coord_label = f"{label}({x:.2f}, {y:.2f})"
            
            ax.plot(x, y, "o", markersize=6, color="magenta", fillstyle="none",label=coord_label)

        # Plot exclusion zones if available
        if 'exclusion_bounds' in map_data and map_data['exclusion_bounds'] is not None:
            exclusion_bounds = map_data['exclusion_bounds']
            
            if exclusion_bounds['type'] == 'phi':
                # Use actual azimuth bounds from the exclusion zone calculation
                azimuth_min = exclusion_bounds['azimuth_min']
                azimuth_max = exclusion_bounds['azimuth_max']
                azimuth_center = exclusion_bounds['azimuth_center']
                                
                ax.axvline(x=azimuth_min, color='red', linestyle='--', alpha=0.7, linewidth=1, label='Exclusion zone')
                ax.axvline(x=azimuth_max, color='red', linestyle='--', alpha=0.7, linewidth=1)

        cbar = fig.colorbar(c, ax=ax)
        cbar.set_label('Correlation')

        # Set axis labels based on coordinate system - use ax.set_xlabel() not plt.xlabel()
        if coord_system == "cylindrical":
            if rec_type == "phiz":
                ax.set_xlabel("Azimuth Angle, $\\phi$ [$^\\circ$]")
                ax.set_ylabel("Depth, z [m]")
            elif rec_type == "rhoz":
                ax.set_xlabel("Distance, $\\rho$ [m]")
                ax.set_ylabel("Depth, z [m]")
        else:  # spherical
            ax.set_xlabel("Azimuth Angle, $\\phi$[$^\\circ$]")
            ax.set_ylabel("Zenith Angle, $\\theta$[$^\\circ$]")

        channels = map_data['channels']
        fixed_coord = map_data['fixed_coord']
        
        if coord_system == "spherical":
            ax.set_title(
                (
                    f"St: {station_id}, run(s) {run_number}, "
                    + f"event: {event_number}, "
                    + f"ch(s): {channels}\n"
                    + f"r $\\equiv$ {fixed_coord}m"
                ),
            )
        else:
            if rec_type == "phiz":
                ax.set_title(
                    (
                        f"St: {station_id}, run(s): {run_number}, "
                        + f"event: {event_number}, "
                        + f"ch(s): {channels}\n"
                        + f"$\\rho\\equiv$ {fixed_coord}m"
                    ),
                )
            else:  # rhoz
                ax.set_title(
                    (
                        f"Station: {station_id}, run(s): {run_number}, "
                        + f"event: {event_number}, "
                        + f"ch's: {channels}, "
                        + f"$\\phi\\equiv$ {fixed_coord}°"
                    ),
                )
        
        if create_minimaps:
            try:
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                
                has_alt = ('coord0_alt' in map_data and map_data['coord0_alt'] is not None and 
                        not np.isnan(map_data['coord0_alt']) and not np.isnan(map_data['coord1_alt']))
                
                if has_alt:
                    # Get alt coordinates and convert to plotting units if needed
                    alt_x_center = map_data['coord0_alt']
                    alt_y_center = map_data['coord1_alt']
                    
                    if coord_system == "cylindrical" and rec_type == "phiz":
                        alt_x_center = np.degrees(alt_x_center)
                    elif coord_system == "spherical":
                        alt_x_center = np.degrees(alt_x_center)
                        alt_y_center = np.degrees(alt_y_center)
                    
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
                    
                    zoom_width = 20   # degrees around peak
                    zoom_height = 10  # meters around peak
                    
                    # Left minimap
                    left_zoom_x_min = left_point_x - zoom_width/2
                    left_zoom_x_max = left_point_x + zoom_width/2
                    left_zoom_y_min = left_point_y - zoom_height/2
                    left_zoom_y_max = left_point_y + zoom_height/2
                    
                    # Position left minimap on center left to avoid overlap with peaks and legend
                    inset_ax_left = inset_axes(ax, width="20%", height="20%", loc='center left', borderpad=1.5)
                    inset_ax_left.imshow(
                        corr_matrix,
                        cmap=mycmap,
                        vmin=np.nanmin(corr_matrix),
                        vmax=np.nanmax(corr_matrix),
                        extent=[limits[0], limits[1], limits[2], limits[3]],
                        origin='lower',
                        interpolation='nearest',
                        aspect='auto',
                        rasterized=True,
                    )
                    
                    if left_is_primary:
                        inset_ax_left.plot(left_point_x, left_point_y, "o", markersize=6, color="lime")
                    else:
                        inset_ax_left.plot(left_point_x, left_point_y, "o", markersize=6, color="lime", fillstyle="none", markeredgewidth=2)
                    
                    # Plot extra points (like true vertex) if they fall within left minimap bounds
                    points_to_plot = extra_points_override if extra_points_override is not None else self.extra_points
                    for pt_x, pt_y, pt_label in points_to_plot:
                        if left_zoom_x_min <= pt_x <= left_zoom_x_max and left_zoom_y_min <= pt_y <= left_zoom_y_max:
                            inset_ax_left.plot(pt_x, pt_y, "o", markersize=6, color="magenta")
                    
                    inset_ax_left.set_xlim(left_zoom_x_min, left_zoom_x_max)
                    inset_ax_left.set_ylim(left_zoom_y_min, left_zoom_y_max)
                    inset_ax_left.tick_params(labelsize=8)
                    for spine in inset_ax_left.spines.values():
                        spine.set_edgecolor('white')
                        spine.set_linewidth(1)
                    
                    # Right minimap
                    right_zoom_x_min = right_point_x - zoom_width/2
                    right_zoom_x_max = right_point_x + zoom_width/2
                    right_zoom_y_min = right_point_y - zoom_height/2
                    right_zoom_y_max = right_point_y + zoom_height/2
                    
                    # Position right minimap on center right to avoid overlap with peaks and legend
                    inset_ax_right = inset_axes(ax, width="20%", height="20%", loc='center right', borderpad=1.5)
                    inset_ax_right.imshow(
                        corr_matrix,
                        cmap=mycmap,
                        vmin=np.nanmin(corr_matrix),
                        vmax=np.nanmax(corr_matrix),
                        extent=[limits[0], limits[1], limits[2], limits[3]],
                        origin='lower',
                        interpolation='nearest',
                        aspect='auto',
                        rasterized=True,
                    )
                    
                    if left_is_primary:
                        inset_ax_right.plot(right_point_x, right_point_y, "o", markersize=6, color="lime", fillstyle="none", markeredgewidth=2)
                    else:
                        inset_ax_right.plot(right_point_x, right_point_y, "o", markersize=6, color="lime")
                    
                    # Plot extra points (like true vertex) if they fall within right minimap bounds
                    points_to_plot = extra_points_override if extra_points_override is not None else self.extra_points
                    for pt_x, pt_y, pt_label in points_to_plot:
                        if right_zoom_x_min <= pt_x <= right_zoom_x_max and right_zoom_y_min <= pt_y <= right_zoom_y_max:
                            inset_ax_right.plot(pt_x, pt_y, "o", markersize=6, color="magenta")
                    
                    inset_ax_right.set_xlim(right_zoom_x_min, right_zoom_x_max)
                    inset_ax_right.set_ylim(right_zoom_y_min, right_zoom_y_max)
                    inset_ax_right.tick_params(labelsize=8)
                    for spine in inset_ax_right.spines.values():
                        spine.set_edgecolor('white')
                        spine.set_linewidth(1)
                    
                else:
                    # Single minimap around primary maximum
                    zoom_width = 25   # degrees around peak
                    zoom_height = 12  # meters around peak
                    
                    zoom_x_min = max_corr_x - zoom_width/2
                    zoom_x_max = max_corr_x + zoom_width/2
                    zoom_y_min = max_corr_y - zoom_height/2
                    zoom_y_max = max_corr_y + zoom_height/2
                    
                    # Position single minimap on center right to avoid overlap with peak and legend
                    inset_ax = inset_axes(ax, width="25%", height="25%", loc='center right', borderpad=1.5)
                    inset_ax.imshow(
                        corr_matrix,
                        cmap=mycmap,
                        vmin=np.nanmin(corr_matrix),
                        vmax=np.nanmax(corr_matrix),
                        extent=[limits[0], limits[1], limits[2], limits[3]],
                        origin='lower',
                        interpolation='nearest',
                        aspect='auto',
                        rasterized=True,
                    )
                    inset_ax.plot(max_corr_x, max_corr_y, "o", markersize=6, color="lime")
                    
                    # Plot extra points (like true vertex) if they fall within single minimap bounds
                    points_to_plot = extra_points_override if extra_points_override is not None else self.extra_points
                    for pt_x, pt_y, pt_label in points_to_plot:
                        if zoom_x_min <= pt_x <= zoom_x_max and zoom_y_min <= pt_y <= zoom_y_max:
                            inset_ax.plot(pt_x, pt_y, "o", markersize=6, color="magenta")
                    
                    inset_ax.set_xlim(zoom_x_min, zoom_x_max)
                    inset_ax.set_ylim(zoom_y_min, zoom_y_max)
                    inset_ax.tick_params(labelsize=8)
                    for spine in inset_ax.spines.values():
                        spine.set_edgecolor('white')
                        spine.set_linewidth(1.5)
                                        
            except Exception as e:
                print(f"Warning: Could not create minimaps: {e}")
        
        #plt.tight_layout()
        ax.legend()
        
        # If standalone mode, save figure and return path
        if standalone:
            output_path = determine_plot_output_path(
                self.map_data_path, self.output_arg, station_id, 
                run_number, event_number
            )
            
            # Insert coordinate system, reconstruction type, and ray type mode into path
            # Before: figures/station21/run70/station21_run70_evt0_corrmap.png
            # After: figures/station21/run70/spherical/phitheta/viscosity/station21_run70_evt0_corrmap.png
            # (or without mode subdirectory if not present in old data or if mode is 'auto')
            base_dir = os.path.dirname(output_path)
            filename = os.path.basename(output_path)
            
            # Build subdirectory path with coord_system and rec_type
            coord_subdir = os.path.join(base_dir, coord_system)
            if rec_type is not None:
                coord_subdir = os.path.join(coord_subdir, rec_type)
            
            # Add ray_type_mode subdirectory if available
            ray_type_mode = map_data.get('ray_type_mode')
            if ray_type_mode is not None:
                # Add subdirectory for all modes including 'auto'
                coord_subdir = os.path.join(coord_subdir, ray_type_mode)
            
            # Create the subdirectory if it doesn't exist
            os.makedirs(coord_subdir, exist_ok=True)
            
            # Build final output path
            output_path = os.path.join(coord_subdir, filename)
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved correlation map plot to {output_path}")
            plt.close()
            
            return output_path
        else:
            # Embedded mode: return the axes for further use
            return ax


    def plot_comprehensive(self, reco_results_file):
        """
        Generate comprehensive event visualization with correlation map, waveforms, and event info.
        
        Parameters
        ----------
        reco_results_file : str
            Path to reconstruction results HDF5 file containing event data and filenames
        
        Returns
        -------
        str
            Path to saved plot file
        """
        
        if self.map_data is None:
            print("Error: No correlation map data loaded")
            return None
        
        import h5py
        
        map_data = self.map_data
        station_id = map_data['station_id']
        run_number = map_data['run_number']
        event_number = map_data['event_number']
        coord_system = map_data['coord_system']
        rec_type = map_data.get('rec_type')
        
        # Load reconstruction results and extract data filename
        if not reco_results_file.endswith(('.h5', '.hdf5')):
            print(f"Error: Expected HDF5 file, got {reco_results_file}")
            return None
        
        data_filename = None
        try:
            with h5py.File(reco_results_file, 'r') as f:
                results_group = f['results']
                
                # Get arrays from HDF5
                reco_run_numbers = results_group['runNum'][:]
                reco_event_numbers = results_group['eventNum'][:]
                filenames = results_group['filename'][:]
                
                # Find the row matching our run_number and event_number
                matches = np.where((reco_run_numbers == run_number) & (reco_event_numbers == event_number))[0]
                
                if len(matches) == 0:
                    print(f"Error: No matching event in HDF5 file (station {station_id}, run {run_number}, event {event_number})")
                    return None
                
                # Get the first match
                match_idx = matches[0]
                data_filename = filenames[match_idx]
                
                # Handle bytes vs string
                if isinstance(data_filename, bytes):
                    data_filename = data_filename.decode('utf-8')
        
        except Exception as e:
            print(f"Error reading from HDF5 file: {e}")
            return None
        
        if data_filename is None:
            print(f"Error: Could not extract data filename from HDF5")
            return None
        
        # Create figure with GridSpec for flexible layout (3 rows)
        fig = plt.figure(figsize=(18, 16))
        gs = GridSpec(3, 2, figure=fig, height_ratios=[1.5, 0.35, 1.2], width_ratios=[1, 1],
                    hspace=0.25, wspace=0.3)
        
        # Row 1: Correlation map (left) and Ray path visualization (right)
        # Row 2: Event information table (spans both columns)
        # Row 3: Waveform grid (spans both columns)
        
        # Correlation map (top left)
        ax_corr = fig.add_subplot(gs[0, 0])
        
        # Ray path visualization (top right)
        ax_raypath = fig.add_subplot(gs[0, 1])
        
        # Event information table (middle row, spans both columns)
        ax_info = fig.add_subplot(gs[1, :])
        ax_info.axis('off')
        
        # Bottom: Waveform grid (4 cols x 3 rows)
        vpol_channels = [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23]
        
        # Variables to store extracted simulation info
        table_data = []
        table_header = None
        vertex_extra_points = []  # Will hold true vertex location for correlation map
        vertex_position_3d = None  # Will hold 3D vertex position for ray tracing
        
        # Read event and extract both waveforms and metadata in one pass
        if data_filename is not None and data_filename.endswith('.nur'):
            try:
                # Use eventReader to properly read NUR files
                reader = eventReader()
                reader.begin(data_filename)
                
                # Find the matching event
                evt = None
                for event_candidate in reader.run():
                    if event_candidate.get_run_number() == run_number:
                        evt = event_candidate
                        break
                
                if evt is not None:
                    # Initialize detector
                    det = rnog_detector.Detector()
                    det.update(datetime.datetime(2022, 10, 1))
                    
                    station = evt.get_station(station_id)
                    
                    # Extract simulation metadata for info panel
                    if station.has_sim_station():
                        for sim_shower, particle in zip(evt.get_sim_showers(), evt.get_particles()):
                            try:
                                lgE = sim_shower.get_parameter(showerParameters.energy)
                                vertex = sim_shower.get_parameter(showerParameters.vertex)
                                
                                nu_zenith_deg = np.rad2deg(particle.get_parameter(particleParameters.zenith))
                                nu_azimuth_deg = np.rad2deg(particle.get_parameter(particleParameters.azimuth))
                                
                                # Convert to relative PA frame
                                station_pos_abs = np.array(det.get_absolute_position(station_id))
                                ch1_pos_rel = np.array(det.get_relative_position(station_id, 1))
                                ch2_pos_rel = np.array(det.get_relative_position(station_id, 2))
                                pa_pos_rel_station = 0.5 * (ch1_pos_rel + ch2_pos_rel)
                                
                                pa_pos_rel_surf = pa_pos_rel_station + [0, 0, station_pos_abs[2]]
                                
                                vertex_rel_pa = np.array(vertex) - station_pos_abs - pa_pos_rel_station
                                x, y, z = vertex_rel_pa
                                r = np.sqrt(x**2 + y**2 + z**2)
                                rho = np.sqrt(x**2 + y**2)
                                zen_rad = np.arccos(z / r) if r > 0 else 0
                                az_rad = np.arctan2(y, x)
                                zen_deg = np.degrees(zen_rad)
                                az_deg = np.degrees(az_rad) % 360
                                
                                # Store 3D vertex position for ray path plotting
                                print(f"vertex: {np.array(vertex)}")
                                vertex_position_3d = np.array(vertex) - station_pos_abs
                                vertex_position_3d[2] = np.array(vertex)[2] # correct for station offset issue
                                
                                vertex_position_2d = vertex_position_3d
                                vertex_position_2d[0] = np.sqrt(vertex_position_3d[0]**2 + vertex_position_3d[1]**2)
                                vertex_position_2d[1] = 0
                                                                 
                                # Build table data - horizontal format with 4 columns (2x4 table)
                                table_header = os.path.basename(data_filename)
                                # Two rows: headers and values
                                table_data = [
                                    ['Energy (lgE)', 'Vertex X', 'Vertex Y', 'Vertex Z', 'Vertex R', 'Vertex Rho', 'Zenith', 'Azimuth'],
                                    [f'$10^{{{np.log10(lgE):.2f}}}$', f'{x:.2f} m', f'{y:.2f} m', f'{z:.2f} m',
                                     f'{r:.2f} m', f'{rho:.2f} m', f'{nu_zenith_deg:.1f}°', f'{nu_azimuth_deg:.1f}°']
                                ]
                                
                                # Determine true vertex location for correlation map based on coord system
                                if coord_system == "spherical":
                                    # Spherical: use azimuth and zenith directly
                                    vertex_extra_points = [(az_deg, zen_deg, "True Vertex: ")]
                                elif coord_system == "cylindrical":
                                    # Transform z coordinate relative to PA
                                    z_transformed = np.array(vertex)[2]
                                    if rec_type == "phiz":
                                        # Cylindrical phi-z: use azimuth and transformed z
                                        vertex_extra_points = [(az_deg, z_transformed, "True Vertex: ")]
                                    elif rec_type == "rhoz":
                                        # Cylindrical rho-z: use rho and transformed z
                                        vertex_extra_points = [(rho, z_transformed, "True Vertex: ")]
                                
                            except Exception as e:
                                print(f"Warning: Could not extract shower parameters: {e}")
                                table_data = [['Error', 'Could not extract parameters']]
                            break
                    
                    # Apply processing modules (same as interferometric_reco_example.py)
                    # Get config from correlation map for processing settings
                    config = map_data.get('config', {})
                    
                    # Initialize processing modules
                    resampler = channelResampler()
                    resampler.begin()
                    
                    bandpass = channelBandPassFilter()
                    bandpass.begin()
                    
                    channel_add_cable_delay = channelAddCableDelay()
                    channel_add_cable_delay.begin()
                    
                    cw_filter = channelSinewaveSubtraction()
                    cw_filter.begin(save_filtered_freqs=False, freq_band=(0.1, 0.7))
                    
                    # Apply upsampling if configured
                    if config.get('apply_upsampling', False):
                        resampler.run(evt, station, det, sampling_rate=5 * units.GHz)
                    
                    # Apply bandpass filter if configured
                    if config.get('apply_bandpass', False):
                        bandpass.run(evt, station, det, 
                                   passband=[0.1 * units.GHz, 0.6 * units.GHz],
                                   filter_type='butter', order=10)
                    
                    # Apply CW removal if configured
                    if config.get('apply_cw_removal', False):
                        peak_prominence = config.get('cw_peak_prominence', 4.0)
                        cw_filter.run(evt, station, det, peak_prominence=peak_prominence)
                    
                    # Apply cable delays (subtract for simulation -> data conversion)
                    channel_add_cable_delay.run(evt, station, det, mode='subtract')
                    
                    # Extract processed waveforms
                    waveform_data = {}
                    for ch_id in vpol_channels:
                        if not station.has_channel(ch_id):
                            continue
                        
                        channel = station.get_channel(ch_id)
                        trace = channel.get_trace()
                        times = channel.get_times()
                        
                        waveform_data[ch_id] = {
                            'times': times / units.ns,
                            'trace': trace / units.mV,
                        }
                    
                    # Create waveform grid (4 cols x 3 rows for up to 12 channels)
                    # Position it in the bottom third of the figure
                    wf_gs = GridSpec(3, 4, figure=fig, left=0.05, right=0.98, bottom=0.05, top=0.4,
                                   hspace=0.45, wspace=0.35)
                                        
                    for idx, ch_id in enumerate(vpol_channels):
                        if idx >= 12:
                            break
                        
                        row = idx // 4
                        col = idx % 4
                        ax_wf = fig.add_subplot(wf_gs[row, col])
                        
                        if ch_id in waveform_data:
                            data = waveform_data[ch_id]
                            ax_wf.plot(data['times'], data['trace'], linewidth=0.7, color='blue', label=f'Ch {ch_id}')
                            #ax_wf.set_title(f'Ch {ch_id}', fontsize=10, fontweight='bold')
                            ax_wf.grid(True, alpha=0.3)
                            ax_wf.legend()
                            ax_wf.set_ylabel('V [mV]')
                            if row == 2:  # Bottom row
                                ax_wf.set_xlabel('Time [ns]')
                        else:
                            ax_wf.text(0.5, 0.5, f'Ch {ch_id}\n(no data)', ha='center', va='center',
                                    transform=ax_wf.transAxes)
                            ax_wf.set_title(f'Ch {ch_id}', fontweight='bold')
                            ax_wf.grid(True, alpha=0.3)
                    
                else:
                    print(f"Warning: Could not find event with run_number {run_number}")
                    ax_wf = fig.add_subplot(gs[2, :])
                    ax_wf.text(0.5, 0.5, f'Event not found in NUR file', 
                            ha='center', va='center', transform=ax_wf.transAxes)
                    ax_wf.axis('off')
                
                reader.end()
            
            except Exception as e:
                print(f"Warning: Could not process NUR file: {e}")
                import traceback
                traceback.print_exc()
                ax_wf = fig.add_subplot(gs[2, :])
                ax_wf.text(0.5, 0.5, f'Could not process waveforms: {e}', 
                        ha='center', va='center', transform=ax_wf.transAxes)
                ax_wf.axis('off')
        else:
            # No NUR file provided or file is HDF5, show placeholder
            ax_wf = fig.add_subplot(gs[2, :])
            ax_wf.text(0.5, 0.5, 'Waveforms unavailable (NUR file required)', 
                    ha='center', va='center', transform=ax_wf.transAxes)
            ax_wf.axis('off')
        
        # Now plot correlation map with true vertex location
        # Pass force_minimaps to enable minimaps even in embedded mode if user requested them
        self.plot_correlation_map(ax=ax_corr, fig=fig, standalone=False, 
                                 extra_points_override=vertex_extra_points, 
                                 force_minimaps=self.show_minimaps)
        ax_corr.set_title("Correlation Map", fontweight='bold')
        
        # Plot ray paths if we have vertex information
        if vertex_position_3d is not None:
            try:
                plot_ray_paths(pa_pos_rel_surf, vertex_position_2d, nu_zenith_deg=nu_zenith_deg, ax=ax_raypath, 
                             ice_model='greenland_simple', 
                             initial_label='PA Center',
                             final_label='Vertex',
                             show_legend=True)
                ax_raypath.set_title('Ray Paths: Vertex → PA', fontweight='bold')
            except Exception as e:
                print(f"Warning: Could not plot ray paths: {e}")
                import traceback
                traceback.print_exc()
                ax_raypath.text(0.5, 0.5, f'Could not plot ray paths\n{e}', 
                              ha='center', va='center', transform=ax_raypath.transAxes, wrap=True)
                ax_raypath.axis('off')
        else:
            # No vertex information available
            ax_raypath.text(0.5, 0.5, 'Ray path unavailable\n(No vertex information)', 
                          ha='center', va='center', transform=ax_raypath.transAxes)
            ax_raypath.axis('off')
        
        # Create nicely formatted horizontal table in info panel
        if table_data:
            # Add header title first
            if table_header:
                ax_info.text(0.5, 0.88, table_header, transform=ax_info.transAxes,
                           verticalalignment='top', ha='center', 
                           family='monospace', weight='bold',
                           bbox=dict(boxstyle='round', facecolor='#E8F4F8', alpha=0.8))
            
            # Create horizontal table (2 rows: headers and values)
            table = ax_info.table(cellText=table_data, cellLoc='center', loc='center',
                                 bbox=[0.05, 0.05, 0.85, 0.65])
            
            #table.auto_set_font_size(False)
            #table.set_fontsize(9)
            
            # Style the table
            for (i, j), cell in table.get_celld().items():
                if i == 0:  # Header row
                    cell.set_text_props(weight='bold', ha='center')
                    cell.set_facecolor('#E8F4F8')
                else:  # Value row
                    cell.set_text_props(ha='center', family='monospace')
                    cell.set_facecolor('#FFFFFF')
                
                cell.set_edgecolor('#CCCCCC')
                cell.set_linewidth(0.5)
        else:
            # Fallback to simple text if no table data
            info_text = f"Station {station_id} | Run {run_number} | Event {event_number}"
            ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Overall title - use same format as plot_correlation_map
        channels = map_data['channels']
        fixed_coord = map_data['fixed_coord']
        
        if coord_system == "spherical":
            title_text = (
                f"St: {station_id}, run(s) {run_number}, "
                f"event: {event_number}, "
                f"ch(s): {channels}\n"
                f"r $\\equiv$ {fixed_coord}m"
            )
        else:
            if rec_type == "phiz":
                title_text = (
                    f"St: {station_id}, run(s): {run_number}, "
                    f"event: {event_number}, "
                    f"ch(s): {channels}\n"
                    f"$\\rho\\equiv$ {fixed_coord}m"
                )
            else:  # rhoz
                title_text = (
                    f"Station: {station_id}, run(s): {run_number}, "
                    f"event: {event_number}, "
                    f"ch's: {channels}, "
                    f"$\\phi\\equiv$ {fixed_coord}°"
                )
        
        fig.suptitle(title_text, fontweight='bold', y=0.99)
        
        # Determine output path
        output_path = determine_plot_output_path(
            self.map_data_path, self.output_arg, station_id, 
            run_number, event_number
        )
        
        # Insert coordinate system, reconstruction type, and ray type mode into path
        # Before: figures/station21/run70/station21_run70_evt0_corrmap.png
        # After: figures/station21/run70/spherical/phitheta/viscosity/station21_run70_evt0_corrmap.png
        # (or without mode subdirectory if not present in old data or if mode is 'auto')
        base_dir = os.path.dirname(output_path)
        filename = os.path.basename(output_path)
        
        # Build subdirectory path with coord_system and rec_type
        coord_subdir = os.path.join(base_dir, coord_system)
        if rec_type is not None:
            coord_subdir = os.path.join(coord_subdir, rec_type)
        
        # Add ray_type_mode subdirectory if available
        ray_type_mode = map_data.get('ray_type_mode')
        if ray_type_mode is not None:
            # Add subdirectory for all modes including 'auto'
            coord_subdir = os.path.join(coord_subdir, ray_type_mode)
        
        # Create the subdirectory if it doesn't exist
        os.makedirs(coord_subdir, exist_ok=True)
        
        # Build final output path
        output_path = os.path.join(coord_subdir, filename)
        
        # Change extension to _comprehensive.png
        output_path = output_path.replace('.png', '_comprehensive.png')
        
        # Save with 25% higher DPI to make everything 25% larger (150 * 1.25 = 187.5)
        plt.savefig(output_path, dpi=187.5)
        print(f"Saved comprehensive plot to {output_path}")
        plt.close()
        
        # Create separate per-channel ray path plot if we have vertex information
        if vertex_position_3d is not None:
            try:
                per_channel_output_path = output_path.replace('_comprehensive.png', '_ray_paths_per_channel.png')
                
                # Reformat waveform_data for per-channel function (dict of ch_id: (times, voltages))
                wf_data_simple = None
                if 'waveform_data' in locals() and waveform_data:
                    wf_data_simple = {ch: (data['times'], data['trace']) 
                                     for ch, data in waveform_data.items()}
                
                self._plot_per_channel_ray_paths(
                    pa_pos_rel_surf, vertex_position_2d, nu_zenith_deg, 
                    det, station_id, vpol_channels,
                    per_channel_output_path, wf_data_simple
                )
                print(f"Saved per-channel ray paths to {per_channel_output_path}")
            except Exception as e:
                print(f"Warning: Could not create per-channel ray path plot: {e}")
                import traceback
                traceback.print_exc()
        
        return output_path
    
    def _plot_per_channel_ray_paths(self, pa_pos_rel_surf, vertex_position_2d, nu_zenith_deg, 
                                    det, station_id, channels, output_path, waveform_data=None):
        """
        Create a grid plot showing ray paths from vertex to each individual channel,
        with optional waveforms below each ray path.
        
        Parameters
        ----------
        pa_pos_rel_surf : array
            PA position relative to surface
        vertex_position_2d : array
            Vertex position in 2D
        nu_zenith_deg : float
            Zenith angle in degrees
        det : Detector
            Detector object
        station_id : int
            Station ID
        channels : list
            List of channel IDs to plot
        output_path : str
            Path to save the output plot
        waveform_data : dict, optional
            Dictionary mapping channel IDs to (times, voltages) tuples
        """
        # Determine if we're plotting waveforms
        plot_waveforms = waveform_data is not None and len(waveform_data) > 0
        
        # Create figure with appropriate size and grid
        if plot_waveforms:
            # 4 rows alternating: ray path, waveform, ray path, waveform + 1 legend column
            fig = plt.figure(figsize=(32, 16))
            gs = plt.GridSpec(4, 7, figure=fig, width_ratios=[1, 1, 1, 1, 1, 1, 0.15], 
                             height_ratios=[1, 0.6, 1, 0.6],
                             hspace=0.25, wspace=0.35, left=0.04, right=0.96, top=0.96, bottom=0.05)
        else:
            # Original: 2 rows for ray paths + 1 legend column
            fig = plt.figure(figsize=(26, 8))
            gs = plt.GridSpec(2, 7, figure=fig, width_ratios=[1, 1, 1, 1, 1, 1, 0.15], 
                             hspace=0.3, wspace=0.35, left=0.04, right=0.96)
        
        # Get station absolute position for coordinate conversion
        station_pos_abs = np.array(det.get_absolute_position(station_id))
        
        # Keep track of handles and labels for shared legend
        all_handles = []
        all_labels = []
        handles_collected = False
        
        for idx, ch_id in enumerate(channels):
            if idx >= 12:  # Safety check
                break
            
            # Alternating layout: first 6 channels in row 0, next 6 in row 2
            row = 0 if idx < 6 else 2
            col = idx % 6
            ax = fig.add_subplot(gs[row, col])
            
            # Get channel position relative to station
            ch_pos_rel_station = np.array(det.get_relative_position(station_id, ch_id))
            
            # Convert to surface-relative coordinates for ray tracing
            # Channel position absolute
            ch_pos_abs = ch_pos_rel_station + [0, 0, station_pos_abs[2]]
            
            # For ray tracing, we need the channel position in 2D (rho, z)
            ch_pos_2d = np.array([
                np.sqrt(ch_pos_rel_station[0]**2 + ch_pos_rel_station[1]**2),  # rho
                ch_pos_abs[2]  # z (absolute depth)
            ])
            
            try:
                # Plot ray paths from vertex to this specific channel
                # Build labels with actual (R, Z) coordinates used in ray tracing
                ch_r = ch_pos_2d[0]  # rho (horizontal distance)
                ch_z = ch_pos_2d[1]  # z (depth)
                vertex_r = vertex_position_2d[0]
                vertex_z = vertex_position_2d[1]
                
                plot_ray_paths(
                    ch_pos_abs, vertex_position_2d, nu_zenith_deg,
                    ax=ax, ice_model='greenland_simple',
                    initial_label=f'Ch {ch_id}: ({ch_r:.1f}m, {ch_z:.1f}m)',
                    final_label=f'Vertex: ({vertex_r:.1f}m, {vertex_z:.1f}m)',
                    show_legend=True  # Show legend on each plot for position info
                )
                ax.set_title(f'Ch {ch_id}', fontweight='bold', fontsize=14)
                
                # Collect legend handles and labels from first plot only
                if not handles_collected:
                    handles, labels = ax.get_legend_handles_labels()
                    all_handles = handles
                    all_labels = labels
                    handles_collected = True
                
                # Move legend to top center with single column layout
                legend = ax.get_legend()
                if legend:
                    # Filter to only show vertex and channel position in individual plot legends
                    handles, labels = ax.get_legend_handles_labels()
                    filtered_handles = []
                    filtered_labels = []
                    for h, l in zip(handles, labels):
                        # Keep only Ch and Vertex labels (position info)
                        if l.startswith('Ch ') or l.startswith('Vertex'):
                            filtered_handles.append(h)
                            filtered_labels.append(l)
                    
                    # Remove old legend and create new one at top center
                    legend.remove()
                    ax.legend(filtered_handles, filtered_labels, loc='upper center', 
                             ncol=1, fontsize=10, framealpha=0.9)
                
                # Remove x-label for row 0 (it has waveforms below)
                if row == 0:
                    ax.set_xlabel('')
                
                # Remove y-label for non-leftmost columns
                if col != 0:
                    ax.set_ylabel('')
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Ch {ch_id}\nError: {str(e)[:30]}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title(f'Ch {ch_id}', fontweight='bold', fontsize=14)
        
        # Hide any unused ray path subplots
        for idx in range(len(channels), 12):
            row = 0 if idx < 6 else 2
            col = idx % 6
            ax = fig.add_subplot(gs[row, col])
            ax.axis('off')
        
        # Plot waveforms if provided
        if plot_waveforms:
            for idx, ch_id in enumerate(channels):
                if idx >= 12:  # Maximum 12 channels
                    break
                
                # Waveforms go in rows 1 and 3 (directly below ray paths in rows 0 and 2)
                row = 1 if idx < 6 else 3
                col = idx % 6
                ax_wf = fig.add_subplot(gs[row, col])
                
                if ch_id in waveform_data:
                    times, voltages = waveform_data[ch_id]
                    ax_wf.plot(times, voltages, linewidth=0.8, color='royalblue')
                    ax_wf.set_xlabel('Time [ns]', fontsize=10)
                    ax_wf.set_ylabel('Voltage [mV]', fontsize=10)
                    ax_wf.tick_params(labelsize=9)
                    ax_wf.grid(True, alpha=0.3, linewidth=0.5)
                    
                    # Remove x-label for row 1 (it has ray paths below)
                    if row == 1:
                        ax_wf.set_xlabel('')
                    
                    # Remove y-label for non-leftmost columns
                    if col != 0:
                        ax_wf.set_ylabel('')
                else:
                    ax_wf.text(0.5, 0.5, f'No waveform\nfor Ch {ch_id}', 
                             ha='center', va='center', transform=ax_wf.transAxes)
                    ax_wf.axis('off')
            
            # Hide any unused waveform subplots
            for idx in range(len(channels), 12):
                row = 1 if idx < 6 else 3
                col = idx % 6
                ax_wf = fig.add_subplot(gs[row, col])
                ax_wf.axis('off')
        
        # Create shared legend in empty subplot area (bottom right when plotting waveforms)
        # Only include ray types (direct, reflected) - not channel positions
        if all_handles:
            # Filter to only show ray type labels (direct, reflected, refracted)
            legend_handles = []
            legend_labels = []
            for h, l in zip(all_handles, all_labels):
                # Keep only ray type labels (not channel or vertex position labels)
                if not l.startswith('Ch ') and not l.startswith('Vertex'):
                    legend_handles.append(h)
                    legend_labels.append(l)
            
            # Create legend axes in appropriate location
            if plot_waveforms:
                # Place legend in empty bottom-right area (col 5, rows 2-3)
                legend_ax = fig.add_subplot(gs[2:4, 5])
            else:
                # Original position: rightmost column
                legend_ax = fig.add_subplot(gs[:, 6])
            legend_ax.axis('off')
            legend_ax.legend(legend_handles, legend_labels, loc='center', 
                           fontsize=14, framealpha=0.9, title='Ray Types', title_fontsize=16)
        
        title_text = f'Ray Paths: Vertex → Individual Channels (Station {station_id})'
        fig.suptitle(title_text, fontsize=18, fontweight='bold', y=0.98)
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path

    def plot_pair_correlation_grid(self, pair_map_dir, output_path=None):
        """
        Create a triangular grid plot of individual channel pair correlation maps.
        
        Reads pairwise correlation map pickle files from a directory and arranges them
        in a triangular grid where row/column indices correspond to channel numbers,
        and off-diagonal elements show the correlation maps for those channel pairs.
        
        Parameters
        ----------
        pair_map_dir : str
            Directory containing pairwise correlation map pickle files
            (typically results/station{ID}/run{NUM}/corr_map_data/pairwise_maps/)
        output_path : str, optional
            Path to save the output plot. If None, auto-determines based on first map loaded.
        
        Returns
        -------
        str
            Path to saved plot file
        """
        import glob
        
        # Find all pair correlation map files
        pair_files = glob.glob(os.path.join(pair_map_dir, "*.pkl"))
        
        if not pair_files:
            print(f"Error: No correlation map files found in {pair_map_dir}")
            return None
        
        print(f"Found {len(pair_files)} pairwise correlation maps")
        
        # Load all pair maps and extract channel information
        pair_data = []
        for pkl_file in pair_files:
            try:
                map_data = load_correlation_map(pkl_file)
                
                # Check if this is a pairwise map (has 'pair_channels' key)
                if 'pair_channels' in map_data and map_data['pair_channels'] is not None:
                    pair_channels = map_data['pair_channels']
                    if len(pair_channels) == 2:
                        ch1, ch2 = sorted(pair_channels)
                        pair_data.append({
                            'ch1': ch1,
                            'ch2': ch2,
                            'map_data': map_data,
                            'file': pkl_file
                        })
                else:
                    # Fallback: try to extract from filename
                    filename = os.path.basename(pkl_file)
                    match = re.search(r'_ch(\d+)-ch(\d+)_corrmap\.pkl', filename)
                    if match:
                        ch1, ch2 = int(match.group(1)), int(match.group(2))
                        pair_data.append({
                            'ch1': ch1,
                            'ch2': ch2,
                            'map_data': map_data,
                            'file': pkl_file
                        })
            except Exception as e:
                print(f"Warning: Could not load {pkl_file}: {e}")
        
        if not pair_data:
            print("Error: No valid pair correlation maps found")
            return None
        
        # Get unique channels and create mapping
        all_channels = set()
        for pd in pair_data:
            all_channels.add(pd['ch1'])
            all_channels.add(pd['ch2'])
        channels = sorted(list(all_channels))
        n_channels = len(channels)
        ch_to_idx = {ch: i for i, ch in enumerate(channels)}
        
        # Extract metadata from first map
        first_map = pair_data[0]['map_data']
        station_id = first_map['station_id']
        run_number = first_map['run_number']
        event_number = first_map['event_number']
        coord_system = first_map['coord_system']
        rec_type = first_map.get('rec_type')
        limits = first_map['limits']
        step_sizes = first_map['step_sizes']
        
        # Split into multiple grids if more than 8 channels (to keep max 8x8 = 64 subplots)
        max_channels_per_grid = 8
        if n_channels > max_channels_per_grid:
            print(f"Splitting {n_channels} channels into multiple {max_channels_per_grid}x{max_channels_per_grid} grids")
            n_grids = int(np.ceil(n_channels / max_channels_per_grid))
            output_paths = []
            
            for grid_idx in range(n_grids):
                start_idx = grid_idx * max_channels_per_grid
                end_idx = min((grid_idx + 1) * max_channels_per_grid, n_channels)
                grid_channels = channels[start_idx:end_idx]
                
                # Create subset of pair_data for this grid
                grid_pair_data = [pd for pd in pair_data if pd['ch1'] in grid_channels and pd['ch2'] in grid_channels]
                
                # Generate grid-specific output path
                if output_path is None:
                    output_dir = os.path.join("figures", f"station{station_id}", f"run{run_number}")
                    if coord_system and rec_type:
                        output_dir = os.path.join(output_dir, coord_system, rec_type)
                    
                    # Add ray_type_mode subdirectory if available
                    ray_type_mode = first_map.get('ray_type_mode')
                    if ray_type_mode is not None:
                        output_dir = os.path.join(output_dir, ray_type_mode)
                    
                    os.makedirs(output_dir, exist_ok=True)
                    grid_output_path = os.path.join(output_dir, 
                                          f"station{station_id}_run{run_number}_evt{event_number}_pair_grid_{grid_idx+1}of{n_grids}.png")
                else:
                    # User specified output path - add grid index
                    base, ext = os.path.splitext(output_path)
                    grid_output_path = f"{base}_{grid_idx+1}of{n_grids}{ext}"
                
                # Plot this grid
                self._plot_single_pair_grid(grid_channels, grid_pair_data, station_id, run_number, event_number,
                                           coord_system, rec_type, limits, step_sizes, grid_output_path,
                                           grid_label=f" (Grid {grid_idx+1}/{n_grids})")
                output_paths.append(grid_output_path)
            
            print(f"Created {len(output_paths)} pair correlation grids")
            return output_paths
        
        # Single grid case (≤6 channels)
        if output_path is None:
            output_dir = os.path.join("figures", f"station{station_id}", f"run{run_number}")
            if coord_system and rec_type:
                output_dir = os.path.join(output_dir, coord_system, rec_type)
            
            # Add ray_type_mode subdirectory if available
            ray_type_mode = first_map.get('ray_type_mode')
            if ray_type_mode is not None:
                output_dir = os.path.join(output_dir, ray_type_mode)
            
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 
                                      f"station{station_id}_run{run_number}_evt{event_number}_pair_grid.png")
        
        return self._plot_single_pair_grid(channels, pair_data, station_id, run_number, event_number,
                                          coord_system, rec_type, limits, step_sizes, output_path)
    
    def _plot_single_pair_grid(self, channels, pair_data, station_id, run_number, event_number,
                              coord_system, rec_type, limits, step_sizes, output_path, grid_label=""):
        
        """
        Plot a single pair correlation grid for a subset of channels.
        """
        n_channels = len(channels)
        
        # Create figure with grid of subplots
        fig_size = max(3 * n_channels, 12)
        fig, axes = plt.subplots(n_channels, n_channels, figsize=(fig_size, fig_size))
        
        # Handle cases with few channels
        if n_channels == 1:
            axes = np.array([[axes]])
        elif n_channels == 2:
            axes = np.array(axes).reshape(2, 2)
        
        # Create mapping from channel pairs to correlation matrices
        pair_to_data = {(pd['ch1'], pd['ch2']): pd for pd in pair_data}
        
        # Color map for plots
        mycmap = plt.get_cmap("RdBu_r")
        mycmap.set_bad(color='black')
        
        # Fill the grid
        for i, ch_i in enumerate(channels):
            for j, ch_j in enumerate(channels):
                ax = axes[i, j]
                
                if i == j:
                    # Diagonal: just show channel number
                    ax.text(0.5, 0.5, f'Ch {ch_i}', ha='center', va='center', fontweight='bold', transform=ax.transAxes)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.axis('off')
                    
                elif i < j:
                    # Upper triangle: leave blank
                    ax.axis('off')
                    
                else:
                    # Lower triangle: show correlation map for this pair
                    pair_key = tuple(sorted([ch_i, ch_j]))
                    
                    if pair_key in pair_to_data:
                        pd = pair_to_data[pair_key]
                        corr_matrix = pd['map_data']['corr_matrix']
                        
                        dx, dy = step_sizes[0], step_sizes[1]
                        extent = [limits[0]-dx/2, limits[1]+dx/2, limits[2]-dy/2, limits[3]+dy/2]
                        c = ax.imshow(
                            corr_matrix,
                            cmap=mycmap,
                            vmin=np.nanmin(corr_matrix),
                            vmax=np.nanmax(corr_matrix),
                            extent=extent,
                            origin='lower',
                            interpolation='nearest',
                            aspect='auto',
                            rasterized=True,
                        )

                        max_corr_x = pd['map_data']['coord0']
                        max_corr_y = pd['map_data']['coord1']
                        max_corr_value = pd['map_data'].get('max_corr', np.nan)
                        
                        # Convert to plotting units if needed
                        if coord_system == "cylindrical" and rec_type == "phiz":
                            max_corr_x = np.degrees(max_corr_x)
                        elif coord_system == "spherical":
                            max_corr_x = np.degrees(max_corr_x)
                            max_corr_y = np.degrees(max_corr_y)
                        
                        ax.plot(max_corr_x, max_corr_y, 'o', markersize=4, 
                               color='lime', markeredgecolor='black', markeredgewidth=0.5)
                        
                        # Add colorbar for this subplot
                        cbar = plt.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
                        cbar.ax.tick_params(labelsize=8)
                        
                        # Set title with channel pair and max correlation
                        ax.set_title(f'Ch {ch_j}-{ch_i}\nMax: {max_corr_value:.2f}')
                        
                        # Set axis labels based on position in grid
                        ax.tick_params(labelsize=8)
                        
                        # Only show labels on left column and bottom row
                        if j == 0:
                            if coord_system == "cylindrical":
                                if rec_type == "phiz":
                                    ax.set_ylabel("z [m]")
                                elif rec_type == "rhoz":
                                    ax.set_ylabel("z [m]")
                            else:
                                ax.set_ylabel("θ [°]")
                        else:
                            ax.set_yticklabels([])
                        
                        if i == n_channels - 1:
                            if coord_system == "cylindrical":
                                if rec_type == "phiz":
                                    ax.set_xlabel("φ [°]")
                                elif rec_type == "rhoz":
                                    ax.set_xlabel("ρ [m]")
                            else:
                                ax.set_xlabel("φ [°]")
                        else:
                            ax.set_xticklabels([])
                    else:
                        # No data for this pair
                        ax.text(0.5, 0.5, f'Ch {ch_j}-{ch_i}\n(no data)',
                               ha='center', va='center',
                               transform=ax.transAxes)
                        ax.set_xlim(0, 1)
                        ax.set_ylim(0, 1)
                        ax.axis('off')
        
        # Overall title
        fig.suptitle(f'Channel Pair Correlation Grid{grid_label} - Station {station_id}, Run {run_number}, Event {event_number}',
                    fontweight='bold', y=0.995)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved pair correlation grid to {output_path}")
        plt.close()
        
        return output_path


def main():
    parser = argparse.ArgumentParser(
        prog="correlation_map_plotter.py",
        description="Generate correlation map plots from saved pickle files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Plot map with auto-determined output (creates figures/station21/run476/)
            python correlation_map_plotter.py --input station21_run476_evt7_corrmap.pkl
            
            # Plot all maps in directory with auto-determined outputs
            python correlation_map_plotter.py --input ./results/station21/run476/corr_map_data/
            
            # Plot with custom base directory (creates testing/figures/station21/run476/)  
            python correlation_map_plotter.py --input map.pkl --output testing/
            
            # Plot with specific output file
            python correlation_map_plotter.py --input map.pkl --output custom_plot.png
            
            # Plot maps matching pattern with minimaps enabled
            python correlation_map_plotter.py --input correlation_maps/ --pattern "*station21*" --minimaps
            
            # Create pair correlation grid from pairwise maps directory
            python correlation_map_plotter.py --pair-grid results/station21/run476/corr_map_data/pairwise_maps/
        """
    )
    
    parser.add_argument("--input", "-i", type=str, required=False,
                       help="Input pickle file or directory containing correlation map data")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output directory or specific filename. If not provided, auto-creates 'figures/station{ID}/run{NUM}/' structure. If directory provided, creates organized structure within it.")
    parser.add_argument("--pattern", "-p", type=str, default="*.pkl",
                       help="File pattern to match when input is directory (default: *.pkl)")
    parser.add_argument("--minimaps", action="store_true",
                       help="Create minimap insets showing zoomed-in views around correlation peaks")
    parser.add_argument("--comprehensive", type=str, default=None,
                       help="Path to reconstruction HDF5 file to create comprehensive plots with waveforms")
    parser.add_argument("--pair-grid", type=str, default=None,
                       help="Directory containing pairwise correlation maps to create a channel pair grid plot")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Print detailed processing information")
    parser.add_argument("--extra-points", type=str, nargs="*", default=[],
                       help="Extra points to plot, format: x,y,label (repeat for multiple)")
    
    args = parser.parse_args()
    
    # Handle pair grid plotting mode
    if args.pair_grid:
        if not os.path.isdir(args.pair_grid):
            print(f"Error: --pair-grid path does not exist or is not a directory: {args.pair_grid}")
            return
        
        print(f"Creating pair correlation grid from: {args.pair_grid}")
        plotter = CorrelationMapPlotter()
        output_path = plotter.plot_pair_correlation_grid(args.pair_grid, output_path=args.output)
        
        if output_path:
            print(f"Pair correlation grid saved to: {output_path}")
        else:
            print("Failed to create pair correlation grid")
        
        return
    
    # Validate input argument for normal mode
    if not args.input:
        print("Error: --input argument is required (unless using --pair-grid)")
        return
        
    if os.path.isfile(args.input):
        input_files = [args.input]
    elif os.path.isdir(args.input):
        pattern_path = os.path.join(args.input, args.pattern)
        input_files = glob.glob(pattern_path)
        if not input_files:
            print(f"No files found matching pattern: {pattern_path}")
            return
    else:
        print(f"Input path does not exist: {args.input}")
        return
    
    print(f"Found {len(input_files)} correlation map files to process")
    
    for i, file_path in enumerate(input_files, 1):
        if args.verbose:
            print(f"\nProcessing {i}/{len(input_files)}: {file_path}")
        
        try:
            extra_points = []
            for pt in args.extra_points:
                parts = pt.split(",")
                if len(parts) == 3:
                    try:
                        x, y = float(parts[0]), float(parts[1])
                        label = parts[2]
                        extra_points.append((x, y, label))
                    except Exception:
                        continue
            
            # Create plotter instance
            plotter = CorrelationMapPlotter(
                map_data_path=file_path,
                output_arg=args.output,
                show_minimaps=args.minimaps,
                extra_points=extra_points
            )
            
            if args.comprehensive:
                # Create comprehensive plot with waveforms
                saved_path = plotter.plot_comprehensive(args.comprehensive)
                if args.verbose and saved_path:
                    print(f"  Saved comprehensive plot to: {saved_path}")
            else:
                # Create simple correlation map plot
                saved_path = plotter.plot_correlation_map()
                if args.verbose and saved_path:
                    print(f"  Saved correlation map plot to: {saved_path}")
            
            if args.verbose and plotter.map_data:
                station_id = plotter.map_data['station_id']
                run_number = plotter.map_data['run_number']
                event_number = plotter.map_data['event_number']
                max_corr = np.nanmax(plotter.map_data['corr_matrix'])
                print(f"  Station {station_id}, Run {run_number}, Event {event_number}")
                print(f"  Max correlation: {max_corr:.3f}")
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    print(f"\nCompleted processing {len(input_files)} correlation maps")


if __name__ == "__main__":
    main()