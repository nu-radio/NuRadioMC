import numpy as np
from tools.travel_time_simulator import TravelTimeSimulator
from NuRadioMC.utilities.medium import greenland_simple
import NuRadioReco.detector.detector
from scipy.interpolate import RegularGridInterpolator
import astropy
import argparse
import os
from multiprocessing import Pool


#data_dir = global_constants.DATA_DIR_LOC



data_dir = '/mnt/ceph1-npx/user/anozdrina/rno-g/cr_analysis/' 
deg_step = 1
x_range = np.arange(-200, 201, deg_step)
y_range = np.arange(-200, 201, deg_step)
z_range = np.arange(-100, 1, deg_step)



def get_ant_locs(station, calibrate=0):
    det = NuRadioReco.detector.detector.Detector(
        json_filename="RNO_season_2023.json"
    )
    det.update(astropy.time.Time.now())

    if calibrate:
        cal_locs_file = data_dir + f"station_cal_files/station_{station}.npy"
    else:
        cal_locs_file = None

    channel_ids = [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23]
    if cal_locs_file:
        data = np.load(cal_locs_file)

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
            all_xyz_ant_locs[str(ch)] = det.get_relative_position(
                int(station), int(ch)
            ).tolist()

    return all_xyz_ant_locs


    
def calculate_time_difference(raytracer, ch_0_pos, ch_1_pos, src_posn):
    return raytracer.get_time_differences(ch_0_pos, ch_1_pos, src_posn)


def compute_time_differences(station, ch_pair):
    ant_locs = get_ant_locs(station)

    # Create the meshgrid
    X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing="ij")
    
    # Flatten the meshgrid arrays
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()

    # Combine the flattened arrays into a single array of points
    src_posn_grid = np.vstack((X_flat, Y_flat, Z_flat)).T

    ice_model = greenland_simple()

    ch_0_pos = np.array(ant_locs[str(ch_pair[0])])
    ch_1_pos = np.array(ant_locs[str(ch_pair[1])])

    #num_threads = os.cpu_count()  # Get the number of available CPU cores
    #chunk_size = len(src_posn_grid) // num_threads  # Calculate the chunk size
    # chunk_size = 1
    # chunks = [
    #     src_posn_grid[i : i + chunk_size]
    #     for i in range(0, len(src_posn_grid), chunk_size)
    # ]

    # n_cpus = 4
    # results = Parallel(n_jobs=n_cpus)(
    #     delayed(calculate_time_difference)(ice_model, ch_0_pos, ch_1_pos, chunk)
    #     for chunk in chunks
    # )

    # Flatten the results list
    #time_differences = np.concatenate(results)

    # Reshape the time differences to the shape of the original grid
    #time_differences_3d = time_differences.reshape(X.shape)
    #print(time_differences_3d.shape)
    
    raytracer = TravelTimeSimulator(ice_model)
    # time_difference_matrix = [raytracer.get_time_differences(ch_0_pos, ch_1_pos, src_posn) for src_posn in src_posn_grid]
    
    #print(np.shape(time_difference_matrix))
    
    with Pool(processes=None) as pool:
        # Apply raytracer.get_time_differences to each element in src_posn_grid
        time_difference_matrix = pool.starmap(raytracer.get_time_differences, [(ch_0_pos, ch_1_pos, src_posn) for src_posn in src_posn_grid])
        
    time_difference_matrix = np.array(time_difference_matrix).reshape(X.shape)

    # Define the directory path
    save_path = data_dir + f"interp_tables/station{station}/"
    os.makedirs(save_path, exist_ok=True)  # Create the directory if it doesn't exist

    # Save the file
    np.save(
        save_path + f"station{station}_2023json_time_differences_3d_{ch_pair[0]}_{ch_pair[1]}_200_{deg_step}deg_grid_noraytracing.npy",
        time_difference_matrix,
    )
    print(f'lookup table created: {save_path}station{station}_2023json_time_differences_3d_{ch_pair[0]}_{ch_pair[1]}_200_{deg_step}deg_grid_noraytracing.npy')


def main(station, ch0, ch1):
    compute_time_differences(station, [ch0, ch1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", required=True, type=int)
    parser.add_argument("--ch0", required=True, type=int)
    parser.add_argument("--ch1", required=True, type=int)
    args = parser.parse_args()
    main(args.station, args.ch0, args.ch1)
