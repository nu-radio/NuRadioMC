import numpy as np
import NuRadioMC.SignalProp.analyticraytracing
import NuRadioMC.utilities.medium
import pickle
import argparse

parser = argparse.ArgumentParser(description='Create lookup tables for vertex reconstructor')
parser.add_argument(
    'antenna_depth',
    type=float,
    help='Depth below the ice surface of the antenna in meter'
)
parser.add_argument(
    '--r_min',
    type=float,
    default=10,
    help='Minimum of the horizontal distance from the antenna in the lookup table in meter'
)
parser.add_argument(
    '--r_max',
    type=float,
    default=5000,
    help='Maximum of the horizontal distance from the antenna in the lookup table in meter'
)
parser.add_argument(
    '--z_min',
    type=float,
    default=3000,
    help='Depth up to which the lookup table is calculated in meter'
)
parser.add_argument(
    '--z_max',
    type=float,
    default=50,
    help='Depth up to which the lookup table is calculated in meter'
)
parser.add_argument(
    '--d_r',
    type=float,
    default=2,
    help='Horizontal step size of the lookup table in meter'
)
parser.add_argument(
    '--d_z',
    type=float,
    default=2,
    help='Vertical step size of the lookup table in meter'
)
parser.add_argument(
    '--output_path',
    type=str,
    default='.',
    help='Path to output folder'
)
parser.add_argument(
    '--ice_model',
    type=str,
    default='greenland_simple',
    help='Name of the ice model to be used'
)

if __name__ == "__main__":
    args = parser.parse_args()

    x_pos = np.arange(args.r_min, args.r_max, args.d_r)
    z_pos = np.arange(-args.z_min, -args.z_max, args.d_z)

    ice = NuRadioMC.utilities.medium.get_ice_model(args.ice_model)
    ray_tracing = NuRadioMC.SignalProp.analyticraytracing.ray_tracing_2D(ice)
    channel_types = [{
        'name': 'antenna_{}'.format(args.antenna_depth),
        'z': -1. * args.antenna_depth
    }]

    lookup_table = {
        'header': {
            'x_min': args.r_min,
            'x_max': args.r_max,
            'd_x': args.d_r,
            'z_min': -args.z_min,
            'z_max': -args.z_max,
            'd_z': args.d_z
        }
    }
    for channel_type in channel_types:
        print('Calculating lookup table for ' + channel_type['name'])
        travel_times_direct = np.zeros((len(x_pos), len(z_pos)))
        travel_times_refracted = np.zeros((len(x_pos), len(z_pos)))
        travel_times_reflected = np.zeros((len(x_pos), len(z_pos)))
        for i_x, xx in enumerate(x_pos):
            for i_z, zz in enumerate(z_pos):
                solutions = ray_tracing.find_solutions([xx, zz], [0, channel_type['z']])
                for solution in solutions:
                    if solution['type'] == 1:
                        travel_times_direct[i_x][i_z] = ray_tracing.get_travel_time([xx, zz], [0, channel_type['z']], solution['C0'])
                    if solution['type'] == 2:
                        travel_times_refracted[i_x][i_z] = ray_tracing.get_travel_time([xx, zz], [0, channel_type['z']], solution['C0'])
                    if solution['type'] == 3:
                        travel_times_reflected[i_x][i_z] = ray_tracing.get_travel_time([xx, zz], [0, channel_type['z']], solution['C0'])
        lookup_table[channel_type['name']] = {
            'direct': travel_times_direct,
            'refracted': travel_times_refracted,
            'reflected': travel_times_reflected
        }

    with open('{}/lookup_table_greenland_{:.0f}.p'.format(args.output_path, args.antenna_depth), 'wb') as f:
        pickle.dump(lookup_table, f)
