import numpy as np
import os
from tqdm import tqdm
import multiprocessing
#import global_constants

from NuRadioMC.utilities.medium import greenland_simple, uniform_ice
import NuRadioReco.detector.detector
import astropy.time
from travel_time_simulator_radiopropa import TravelTimeSimulatorRadioPropa
from travel_time_simulator import TravelTimeSimulator


def get_antenna_position(station, ch):
    det = NuRadioReco.detector.detector.Detector("RNO_season_2024.json")
    det.update(astropy.time.Time.now())
    return np.array(det.get_relative_position(station, ch))

def travel_time_gridpoint(i, j, r_vals, z_vals, raytracer, ant_pos):
    r = r_vals[i]
    z = z_vals[j]
    src = ant_pos + np.array([r, 0.0, 0])
    src[2] = z  # set absolute z
    return i, j, raytracer.get_travel_time(src, ant_pos)

def raytrace_arrival_time(raytracer, ant_pos, r_vals, z_vals, num_threads=1):
    from multiprocessing import Pool

    arrival_times = np.full((len(r_vals), len(z_vals)), np.nan)

    with Pool(num_threads) as pool:
        results = pool.starmap(
            travel_time_gridpoint,
            [(i, j, r_vals, z_vals, raytracer, ant_pos) for i in range(len(r_vals)) for j in range(len(z_vals))]
        )

    for i, j, t in results:
        arrival_times[i, j] = t

    return arrival_times

def generate_rz_table(station, ch, rz_res=1, r_max=1600, z_min=-1600, z_max=200, num_threads=1, raytracing=1):
    data_dir = '/mnt/ceph1-npx/user/anozdrina/rno-g/lookup_tables/fixed/' 
    outdir = data_dir + f"interp_tables/station{station}/"
    os.makedirs(outdir, exist_ok=True)  # Create the directory if it doesn't exist
    # outdir = global_constants.REPO_DIR_LOC + f"tools/reconstruction/tables/simple_rz/station{station}/"
    # os.makedirs(outdir, exist_ok=True)

    R_RANGE = np.arange(0, r_max + rz_res, rz_res)
    Z_RANGE = np.arange(z_min, z_max + rz_res, rz_res)

    raytracer = TravelTimeSimulatorRadioPropa(greenland_simple())
    # if raytracing:
    #     raytracer = TravelTimeSimulatorRadioPropa(uniform_ice())
    # else:
    #     raytracer = TravelTimeSimulator(uniform_ice())
        
    ant_pos = get_antenna_position(station, ch)

    print(f"Generating (r,z) travel time map for channel {ch} (station {station}) using {num_threads} threads...")
    times = raytrace_arrival_time(raytracer, ant_pos, R_RANGE, Z_RANGE, num_threads=num_threads)

    outpath = os.path.join(outdir, f"ch{ch}_rz_table_R1_1600Z-1600_200.npz")
    np.savez_compressed(
        outpath,
        r_range_vals=R_RANGE,
        z_range_vals=Z_RANGE,
        data=times
    )
    print(f"Saved to: {outpath}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("station", type=int,  help="Station number")
    parser.add_argument("channel", type=int, help="Channel number")
    parser.add_argument("num_threads", type=int, default=1)
    args = parser.parse_args()

    generate_rz_table(args.station, args.channel, num_threads=args.num_threads)

