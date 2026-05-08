import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import os
import logging

import NuRadioReco.modules.io.coreas.readCoREASDetector
from NuRadioReco.detector.SKA.detector import Detector
from NuRadioReco.modules.io.coreas import coreas
import NuRadioReco.framework.event
from NuRadioReco.utilities import units
from NuRadioReco.utilities import trace_utilities
from NuRadioReco.framework.parameters import electricFieldParameters as efp

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Interpolate CoREAS HDF5 to SKA detector and plot fluence.')
    parser.add_argument('--input_file', type=str,  default="example_data/greenland_starshape_32obs.hdf5", help='Path to the input CoREAS HDF5 file.')
    parser.add_argument('--det_file', type=str, help='Path to the detector .tm or .json file.')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory where figures and HDF5 files will be saved.')
    parser.add_argument('--fig_name', type=str, default=None, help='Base name for the output plots.')
    parser.add_argument('--hdf5_name', type=str, default='interpolated_ska_event', help='Base name for the output HDF5 files.')
    
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    det = Detector(args.det_file, detector_altitude=321600.0 * units.m)
    
    readCoREASDetector = NuRadioReco.modules.io.coreas.readCoREASDetector.readCoREASDetector()
    readCoREASDetector.begin(input_file=args.input_file, interp_lowfreq=30 * units.MHz, interp_highfreq=350 * units.MHz)

    core_positions = [[0, 0]] 

    for i, evt in enumerate(readCoREASDetector.run(det, core_positions, selected_station_channel_ids={0: None})):

        # Optional: Perform any additional NuRadioReco processing here 
        # (e.g., adding hardware response or noise).
        
        evt_id = evt.get_id()
        print(f"Processing Event {evt_id}...")

        if args.fig_name is not None:
            fig, axs = plt.subplot_mosaic([["Fluence_SKA", "cbar"]], figsize=(5, 5), width_ratios=[1, 0.05])
            
            all_ant_x, all_ant_y, all_fluences = [], [], []
    
            for station in evt.get_stations():
                station_id = station.get_id()
                abs_station_pos = det.get_absolute_position(station_id)
                
                sim_station = station.get_sim_station()
                for efield in sim_station.get_electric_fields():
                    rel_pos = efield.get_position()
                    
                    all_ant_x.append(abs_station_pos[0] + rel_pos[0])
                    all_ant_y.append(abs_station_pos[1] + rel_pos[1])
                    
                    fluence = trace_utilities.get_electric_field_energy_fluence(efield.get_trace(), efield.get_times())
                    all_fluences.append(np.sum(fluence))
    
            all_ant_x = np.array(all_ant_x)
            all_ant_y = np.array(all_ant_y)
            all_fluences = np.array(all_fluences)
    
            sc = axs["Fluence_SKA"].scatter(all_ant_x, all_ant_y, c=all_fluences, cmap='plasma', marker='.', s=5)
            axs["Fluence_SKA"].set_aspect('equal')
            axs["Fluence_SKA"].set_xlabel('East [m]')
            axs["Fluence_SKA"].set_ylabel('North [m]')
            axs["Fluence_SKA"].set_title(f'SKA Interpolated Fluence - Event {evt_id}')
            
            axins = axs["Fluence_SKA"].inset_axes([0.05, 0.55, 0.4, 0.4]) 
            axins.scatter(all_ant_x, all_ant_y, c=all_fluences, cmap='plasma', marker='.', s=20)
            
            x1, x2, y1, y2 = -150, -100, -150, -100
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.set_xticks([])
            axins.set_yticks([])
            axins.set_aspect('equal')
            axs["Fluence_SKA"].indicate_inset_zoom(axins, edgecolor="black")
            
            cbar = fig.colorbar(sc, cax=axs["cbar"])
            cbar.set_label('Energy Fluence [eV/m$^2$]')
            
            fig_path = os.path.join(args.output_dir, f"{args.fig_name}_{evt_id}.png")
            plt.savefig(fig_path)
            plt.close(fig)
        
        hdf5_path = os.path.join(args.output_dir, f"{args.hdf5_name}_{evt_id}.hdf5")
        coreas.write_CORSIKA7(evt, hdf5_path)
