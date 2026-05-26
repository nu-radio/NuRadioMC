"""
An example showing how to use the new write_CORSIKA7() function which stores an event as a hdf5 file.
This function is intended to make the sharing of realistic simulations (interpolated to a detector, noise added, triggers added, ...) easier.
Bellow shows an example of a EAS simulation on the star-shape which gets turned into an interpolated event to the detector (the freq. limits are for SKA-Low).
This returns both the new hdf5 file and an example plot showing both the original star-shape simulation and the new interpolated realistic event.
---------------------

Command line input:
    python Interpolated_SKA_Low_event_to_hdf5.py input_file hdf5_name det_file

input_file: str
            station id to be used, default 32
hdf5_name: str
            name for the output hdf5
det_file: str
            path to SKA-Low detector

returns:
    output HDF5 file
    figure showing the star shape and SKA-Low event
"""


import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

import NuRadioReco.modules.io.coreas.readCoREASDetector
from NuRadioReco.detector.SKA.detector import Detector
from NuRadioReco.modules.io.coreas import coreas
from NuRadioReco.utilities import trace_utilities, units

def get_fluence_data(evt, is_reconstructed=False, det=None):
    x, y, fluences = [], [], []
    for station in evt.get_stations():
        # Both Reconstructed HDF5 and Original Star-Shape HDF5 
        # already contain absolute ground coordinates in the e-field.
        if is_reconstructed:
            abs_pos = np.zeros(3)
        else:
            abs_pos = det.get_absolute_position(station.get_id())
            
        sim_station = station.get_sim_station()
        for efield in sim_station.get_electric_fields():
            rel_pos = efield.get_position()
            x.append(abs_pos[0] + rel_pos[0])
            y.append(abs_pos[1] + rel_pos[1])
            f = trace_utilities.get_electric_field_energy_fluence(efield.get_trace(), efield.get_times())
            fluences.append(np.sum(f))
            
    return np.array(x), np.array(y), np.array(fluences), evt.get_id()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Original HDF5 file.')
    parser.add_argument('--hdf5_name', type=str, required=True, help='Reconstructed HDF5 file.')
    parser.add_argument('--det_file', type=str, required=True, help='Detector .tm file.')
    args = parser.parse_args()

    det = Detector(args.det_file)

    # Extract original Star-Shape fluence data
    star_evt = coreas.read_CORSIKA7(args.input_file)
    x_star, y_star, f_star, id_star = get_fluence_data(star_evt, is_reconstructed=True)

    # Interpolate to SKA-Low layout
    readCoREASDetector = NuRadioReco.modules.io.coreas.readCoREASDetector.readCoREASDetector()
    readCoREASDetector.begin(input_file=args.input_file, interp_lowfreq=30 * units.MHz, interp_highfreq=350 * units.MHz)

    core_positions = [[0, 0]] 
    #evt = next(readCoREASDetector.run(det, core_positions))
    # Incase you only want to plot a few stations for speed replace the line before with:
    evt = next(readCoREASDetector.run(det, core_positions, selected_station_channel_ids={station_id: None for station_id in det.get_station_ids()[::50]}))
    
    # Extract interpolated fluence data BEFORE mutating the event
    x_orig, y_orig, f_orig, id_orig = get_fluence_data(evt, is_reconstructed=False, det=det)

    # Write to HDF5
    reconstructed_HDF5 = f"{args.hdf5_name}.hdf5"
    coreas.write_CORSIKA7(evt, reconstructed_HDF5, detector=det)

    # Read back and extract reconstructed fluence data
    reco_evt = coreas.read_CORSIKA7(reconstructed_HDF5)
    x_reco, y_reco, f_reco, id_reco = get_fluence_data(reco_evt, is_reconstructed=True)

    fig, axs = plt.subplot_mosaic([["StarShape", "Interpolated", "Reconstructed", "Cbar"]], 
                                  figsize=(15, 5), width_ratios=[1, 1, 1, 0.05])

    # Normalize color scale across all three datasets to ensure fair comparison
    vmax = max(f_star.max(), f_orig.max(), f_reco.max())
    
    axs["StarShape"].scatter(x_star, y_star, c=f_star, cmap='plasma', marker='o', s=20, vmax=vmax, vmin=0)
    axs["StarShape"].set_title(f"Original Star-Shape")
    axs["StarShape"].set_ylabel("North [m]")

    axs["Interpolated"].scatter(x_orig, y_orig, c=f_orig, cmap='plasma', marker='.', s=5, vmax=vmax, vmin=0)
    axs["Interpolated"].set_title(f"Interpolated SKA-Low")
    
    sc = axs["Reconstructed"].scatter(x_reco, y_reco, c=f_reco, cmap='plasma', marker='.', s=5, vmax=vmax, vmin=0)
    axs["Reconstructed"].set_title(f"Reconstructed SKA-Low")
    
    for key in ["StarShape", "Interpolated", "Reconstructed"]:
        axs[key].set_xlabel("East [m]")

    # The zoomies
    zoom_center_x, zoom_center_y = x_reco[0], y_reco[0]
    x1, x2 = zoom_center_x - 25, zoom_center_x + 25
    y1, y2 = zoom_center_y - 25, zoom_center_y + 25

    axins = axs['Reconstructed'].inset_axes([0.03, 0.55, 0.4, 0.4])
    axins.scatter(x_reco, y_reco, c=f_reco, cmap='plasma', marker='.', s=15, vmax=vmax, vmin=0)
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_aspect('equal')
    axs['Reconstructed'].indicate_inset_zoom(axins, edgecolor="black")

    cbar = fig.colorbar(sc, cax=axs["Cbar"])
    cbar.set_label('Energy Fluence [eV/m$^2$]')

    plt.tight_layout()
    plt.savefig("comparison_footprint.png")
