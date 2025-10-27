#!/usr/bin/env python3
"""
Ray Path Visualization Module

Provides utilities for calculating and plotting ray paths between vertex locations
and detector positions using NuRadioMC's analytic ray tracing.
"""

import numpy as np
import matplotlib.pyplot as plt
from NuRadioMC.SignalProp import propagation
from NuRadioMC.SignalProp.analyticraytracing import solution_types, ray_tracing_2D
from NuRadioMC.utilities import medium
from NuRadioReco.utilities import units


def plot_ray_paths(initial_point, final_point, nu_zenith_deg, ax=None, ice_model='greenland_simple',
                  attenuation_model='GL1', n_reflections=0, initial_label=None, final_label=None,
                  show_legend=True):
    """
    Plot ray paths between two points using NuRadioMC's analytic ray tracing.
    
    This follows the exact approach from the NuRadioMC examples.
    
    Parameters
    ----------
    initial_point : array-like
        3D starting position [x, y, z] in meters
    final_point : array-like
        3D ending position [x, y, z] in meters
    nu_zenith_deg : float
        Zenith angle in degrees
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure (default: None)
    ice_model : str, optional
        Ice model to use (default: 'greenland_simple')
    attenuation_model : str, optional
        Attenuation model to use (default: 'GL1')
    n_reflections : int, optional
        Number of reflections to consider (default: 0)
    initial_label : str, optional
        Label for initial point. If None, auto-generates from coordinates (default: None)
    final_label : str, optional
        Label for final point. If None, auto-generates from coordinates (default: None)
    show_legend : bool, optional
        Whether to show legend on this axes (default: True)
    
    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the plot
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    initial_point = np.array(initial_point) * units.m
    final_point = np.array(final_point) * units.m
    
    prop = propagation.get_propagation_module('analytic')
    ice = medium.get_ice_model(ice_model)

    rays = prop(ice, attenuation_model, n_frequencies_integration=25, n_reflections=n_reflections)
    rays.set_start_and_end_point(initial_point, final_point)
    rays.find_solutions()

    for i in range(rays.get_number_of_solutions()):
        sol_type = solution_types[rays.get_solution_type(i)]
        
        travel_time = rays.get_travel_time(i)
        
        rays2D = ray_tracing_2D(ice, attenuation_model)
        C0 = rays.get_results()[i]['C0']
        xx, zz = rays2D.get_path(
            np.array([initial_point[0], initial_point[2]]),
            np.array([final_point[0], final_point[2]]),
            C0
        )
        
        line, = ax.plot(xx, zz, label=sol_type, linewidth=2)
        
        mid_idx = len(xx) // 2
        text_x = xx[mid_idx]
        text_y = zz[mid_idx] + 5  # 5m above the path
        
        delta_t_ns = travel_time / units.ns
        ax.text(text_x, text_y, f'Δt = {delta_t_ns:.1f} ns', 
               fontsize=10, ha='center', 
               bbox=dict(boxstyle='round,pad=0.3', facecolor=line.get_color(), 
                        alpha=0.7, edgecolor='none'),
               color='white', fontweight='bold')

    if initial_label is None:
        initial_label = f'Initial ({initial_point[0]:.2f}, {initial_point[2]:.2f})'
    if final_label is None:
        final_label = f'Vertex ({final_point[0]:.2f}, {final_point[2]:.2f})'
    
    ax.scatter(initial_point[0], initial_point[2], label=initial_label, zorder=10)
    ax.scatter(final_point[0], final_point[2], label=final_label, zorder=10)
    
    length = 30
    point1_x, point1_y = [final_point[0], final_point[2]]
    point2_x, point2_y = [point1_x + length*np.sin(np.deg2rad(nu_zenith_deg)), point1_y + length*np.cos(np.deg2rad(nu_zenith_deg))]

    ax.plot([point1_x, point2_x], [point1_y, point2_y], color='k')
    
    n_ice = ice.get_index_of_refraction([0,0,final_point[2]])
    cherenkov_angle = np.arccos(1.0 / n_ice)
    new_angle = np.pi/2 - (np.deg2rad(nu_zenith_deg) + cherenkov_angle)
    
    point1_x, point1_y = [final_point[0], final_point[2]]
    point2_x, point2_y = [point1_x - length*np.cos(new_angle), point1_y - length*np.sin(new_angle)]
    ax.plot([point1_x, point2_x], [point1_y, point2_y], color='k')
    
    #lower_ylim = ax.get_ylim()[0]
    lower_ylim = -100
    
    ax.axhspan(lower_ylim, 0, alpha=0.15, color='lightblue', zorder=0)
    
    ax.set_ylim(lower_ylim, 50)
    
    if show_legend:
        ax.legend(loc='upper center', ncol=2, fontsize=14, framealpha=0.9)
    
    ax.set_xlabel('Horizontal Distance [m]')
    ax.set_ylabel('Depth [m]')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    return ax


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot ray paths between vertex and detector')
    parser.add_argument('--vertex', type=float, nargs=3, default=[70, 0, -300],
                       help='Vertex position [x, y, z] in meters')
    parser.add_argument('--detector', type=float, nargs=3, default=[100, 0, -30],
                       help='Detector position [x, y, z] in meters')
    parser.add_argument('--ice-model', type=str, default='greenland_simple',
                       help='Ice model to use')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (if not provided, shows plot)')
    
    args = parser.parse_args()
    
    vertex_pos = np.array(args.vertex)
    detector_pos = np.array(args.detector)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_ray_paths(vertex_pos, detector_pos, ax=ax, ice_model=args.ice_model)
    ax.set_title('Ray Paths: Vertex to Detector')
    
    if args.output:
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"Saved ray path plot to {args.output}")
    else:
        plt.show()
