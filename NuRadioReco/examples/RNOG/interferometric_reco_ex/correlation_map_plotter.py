#!/usr/bin/env python3
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
import numpy as np
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units
from NuRadioReco.utilities.interferometry_io_utilities import (
    load_correlation_map,
    extract_station_run_from_path,
    determine_plot_output_path
)


def plot_correlation_map(map_data, file_path=None, output_arg=None, show_minimaps=False, extra_points=None):
    """
    Generate correlation map plot from saved data.
    
    Parameters
    ----------
    map_data : dict
        Dictionary containing correlation map data
    file_path : str, optional
        Path to original correlation map file (for auto-determining output)
    output_arg : str, optional
        User-provided output argument (directory or file path)
    show_minimaps : bool, optional
        Whether to create minimap insets (default: False)
    
    Returns
    -------
    str
        Path to saved plot file
    """
    
    corr_matrix = map_data['corr_matrix']
    station_id = map_data['station_id']
    run_number = map_data['run_number']
    event_number = map_data['event_number']
    config = map_data['config']
    coord_system = map_data['coord_system']
    rec_type = map_data.get('rec_type')
    limits = map_data['limits']
    
    create_minimaps = show_minimaps
    
    mycmap = plt.get_cmap("RdBu_r")
    mycmap.set_bad(color='black')

    plt.figure(figsize=(12, 8))
    fig, ax = plt.subplots()

    x = np.linspace(limits[0], limits[1], corr_matrix.shape[1] + 1)
    y = np.linspace(limits[2], limits[3], corr_matrix.shape[0] + 1)
    
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
    
    if coord_system == "cylindrical":
        if rec_type == "phiz":
            legend_label = f"Max corr: {max_corr_value:.2f} at ({max_corr_x:.2f}°, {max_corr_y:.2f}m)"
        elif rec_type == "rhoz":
            legend_label = f"Max corr: {max_corr_value:.2f} at ({max_corr_x:.2f}m, {max_corr_y:.2f}m)"
    elif coord_system == "spherical":
        legend_label = f"Max corr: {max_corr_value:.2f} at ({max_corr_x:.2f}°, {(max_corr_y):.2f}°)"
    
    ax.plot(
        max_corr_x,
        max_corr_y,
        "o",
        markersize=5,
        color="lime",
        label=legend_label,
    )
            
    for x, y, label in extra_points:
        ax.plot(x, y, "o", markersize=5, color="magenta", label=label + f"({x}, {y})")

    if 'coord0_alt' in map_data and map_data['coord0_alt'] is not None:
        coord0_alt = map_data['coord0_alt']
        coord1_alt = map_data['coord1_alt']
        
        if not np.isnan(coord0_alt) and not np.isnan(coord1_alt):
            if 'alt_indices' in map_data and map_data['alt_indices'] is not None:
                alt_idx0, alt_idx1 = map_data['alt_indices']
                alt_max_x = x_midpoints[alt_idx0]
                alt_max_y = y_midpoints[alt_idx1]
                alt_corr_val = corr_matrix[alt_idx1, alt_idx0]

                if rec_type == "phiz":
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
                # Convert units for plotting
                if coord_system == "cylindrical" and rec_type == "phiz":
                    coord0_alt_val = coord0_alt / units.deg
                    coord1_alt_val = coord1_alt / units.m
                elif coord_system == "cylindrical" and rec_type == "rhoz":
                    coord0_alt_val = coord0_alt / units.m
                    coord1_alt_val = coord1_alt / units.m
                else:  # spherical
                    coord0_alt_val = coord0_alt / units.deg
                    coord1_alt_val = coord1_alt / units.deg
                
                # Get correlation value at alternate point
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

    # Plot exclusion zones if available
    if 'exclusion_bounds' in map_data and map_data['exclusion_bounds'] is not None:
        exclusion_bounds = map_data['exclusion_bounds']
        if exclusion_bounds['type'] in ['phi', 'rho']:
            coord_step = (limits[1] - limits[0]) / corr_matrix.shape[1]
            exclusion_left = limits[0] + exclusion_bounds['col_start'] * coord_step
            exclusion_right = limits[0] + exclusion_bounds['col_end'] * coord_step
            
            ax.axvline(x=exclusion_left, color='red', linestyle='--', alpha=0.7, linewidth=1, label='Exclusion zone')
            ax.axvline(x=exclusion_right, color='red', linestyle='--', alpha=0.7, linewidth=1)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    cbar = fig.colorbar(c)
    cbar.set_label('Correlation', fontsize=14)

    # Set axis labels based on coordinate system
    if coord_system == "cylindrical":
        if rec_type == "phiz":
            plt.xlabel("Azimuth Angle, $\\phi$ [$^\\circ$]", fontsize=16)
            plt.ylabel("Depth, z [m]", fontsize=16)
        elif rec_type == "rhoz":
            plt.xlabel("Distance, $\\rho$ [m]", fontsize=16)
            plt.ylabel("Depth, z [m]", fontsize=16)
    else:  # spherical
        plt.xlabel("Azimuth Angle, $\\phi$[$^\\circ$]", fontsize=16)
        plt.ylabel("Zenith Angle, $\\theta$[$^\\circ$]", fontsize=16)

    channels = map_data['channels']
    fixed_coord = map_data['fixed_coord']
    
    if coord_system == "spherical":
        plt.title(
            (
                f"St: {station_id}, run(s) {run_number}, "
                + f"event: {event_number}, "
                + f"ch(s): {channels}\n"
                + f"r $\\equiv$ {fixed_coord}m"
            ),
            fontsize=14,
        )
    else:
        if rec_type == "phiz":
            plt.title(
                (
                    f"St: {station_id}, run(s): {run_number}, "
                    + f"event: {event_number}, "
                    + f"ch(s): {channels}\n"
                    + f"$\\rho\\equiv$ {fixed_coord}m"
                ),
                fontsize=14,
            )
        else:  # rhoz
            plt.title(
                (
                    f"Station: {station_id}, run(s): {run_number}, "
                    + f"event: {event_number}, "
                    + f"ch's: {channels}, "
                    + f"$\\phi\\equiv$ {fixed_coord}°"
                ),
                fontsize=14,
            )
    
    if create_minimaps:
        try:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            
            has_alt = ('coord0_alt' in map_data and map_data['coord0_alt'] is not None and 
                      not np.isnan(map_data['coord0_alt']) and not np.isnan(map_data['coord1_alt']))
            
            if has_alt:
                coord0_alt = map_data['coord0_alt']
                coord1_alt = map_data['coord1_alt']
                
                if 'alt_indices' in map_data and map_data['alt_indices'] is not None:
                    alt_idx0, alt_idx1 = map_data['alt_indices']
                    alt_x_center = x_midpoints[alt_idx0]
                    alt_y_center = y_midpoints[alt_idx1]
                else:
                    if coord_system == "cylindrical" and rec_type == "phiz":
                        alt_x_center = coord0_alt / units.deg
                        alt_y_center = coord1_alt / units.m
                    elif coord_system == "cylindrical" and rec_type == "rhoz":
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
                
                zoom_width = 20   # degrees around peak
                zoom_height = 10  # meters around peak
                
                # Left minimap
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
                for spine in inset_ax_left.spines.values():
                    spine.set_edgecolor('white')
                    spine.set_linewidth(1)
                
                # Right minimap
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
                inset_ax.pcolormesh(
                    x_edges, y_edges, corr_matrix,
                    cmap=mycmap, vmin=single_vmin, vmax=single_vmax, rasterized=True
                )
                inset_ax.plot(max_corr_x, max_corr_y, "o", markersize=8, color="lime")
                inset_ax.set_xlim(zoom_x_min, zoom_x_max)
                inset_ax.set_ylim(zoom_y_min, zoom_y_max)
                inset_ax.tick_params(labelsize=8)
                for spine in inset_ax.spines.values():
                    spine.set_edgecolor('white')
                    spine.set_linewidth(1.5)
                                    
        except Exception as e:
            print(f"Warning: Could not create minimaps: {e}")
    
    plt.tight_layout()
    ax.legend()
    
    station_id_from_path, run_number_from_path = None, None
    if file_path:
        station_id_from_path, run_number_from_path = extract_station_run_from_path(file_path)
    
    final_station_id = station_id_from_path if station_id_from_path is not None else station_id
    final_run_number = run_number_from_path if run_number_from_path is not None else run_number
    
    output_path = determine_plot_output_path(file_path, output_arg, final_station_id, final_run_number, event_number)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved correlation map plot to {output_path}")
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
        """
    )
    
    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Input pickle file or directory containing correlation map data")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output directory or specific filename. If not provided, auto-creates 'figures/station{ID}/run{NUM}/' structure. If directory provided, creates organized structure within it.")
    parser.add_argument("--pattern", "-p", type=str, default="*.pkl",
                       help="File pattern to match when input is directory (default: *.pkl)")
    parser.add_argument("--minimaps", action="store_true",
                       help="Create minimap insets showing zoomed-in views around correlation peaks")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Print detailed processing information")
    parser.add_argument("--extra-points", nargs="*", default=[],
        help="Extra points to plot, format: x,y,label (repeat for multiple)")
    
    args = parser.parse_args()
    
    show_minimaps = args.minimaps
    
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
            map_data = load_correlation_map(file_path)
            
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
            
            saved_path = plot_correlation_map(map_data, file_path, args.output, show_minimaps, extra_points)
            
            if args.verbose:
                station_id = map_data['station_id']
                run_number = map_data['run_number']
                event_number = map_data['event_number']
                max_corr = np.nanmax(map_data['corr_matrix'])
                print(f"  Station {station_id}, Run {run_number}, Event {event_number}")
                print(f"  Max correlation: {max_corr:.3f}")
                print(f"  Saved to: {saved_path}")
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    print(f"\nCompleted processing {len(input_files)} correlation maps")


if __name__ == "__main__":
    main()