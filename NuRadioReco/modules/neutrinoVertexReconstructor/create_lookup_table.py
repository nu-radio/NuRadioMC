import numpy as np
import NuRadioMC.SignalProp.analyticraytracing
import NuRadioMC.utilities.medium
import h5py
import argparse
import logging

logger = logging.getLogger('NuRadioReco.create_lookup_table')

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
    antenna_depth = args.antenna_depth

    if antenna_depth != np.round(antenna_depth, 3):
        logger.warning(
            'Antenna depth specified to more than mm precision. Rounding to nearest mm.')
        antenna_depth = np.round(antenna_depth, 3)

    x_pos = np.arange(args.r_min, args.r_max, args.d_r)
    z_pos = np.arange(-args.z_min, -args.z_max, args.d_z)

    ice = NuRadioMC.utilities.medium.get_ice_model(args.ice_model)
    # ray_tracing = NuRadioMC.SignalProp.analyticraytracing.ray_tracing_2D(ice)
    ray_tracing = NuRadioMC.SignalProp.analyticraytracing.ray_tracing(
        ice, config={'propagation':dict(attenuate_ice=False, focusing=False, birefringence=False, focusing_limit=2)})
    channel_types = [{
        'name': 'antenna_{:.3f}'.format(args.antenna_depth),
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
        logger.status('Calculating lookup table for ' + channel_type['name'])
        travel_times_D = np.zeros((len(x_pos), len(z_pos)))
        travel_times_R = np.zeros((len(x_pos), len(z_pos)))
        for i_x, xx in enumerate(x_pos):
            for i_z, zz in enumerate(z_pos):
                ray_tracing.set_start_and_end_point([-xx, 0, zz], [0, 0, channel_type['z']])
                ray_tracing.find_solutions()
                for iS in range(ray_tracing.get_number_of_solutions()):
                    if iS == 0:
                        travel_times_D[i_x][i_z] = ray_tracing.get_travel_time(iS, analytic=True)
                    elif iS == 1:
                        travel_times_R[i_x][i_z] = ray_tracing.get_travel_time(iS, analytic=True)
        lookup_table[channel_type['name']] = {
            'D': travel_times_D,
            'R': travel_times_R,
        }

    # Store lookup tables as HDF5 files
    with h5py.File(f'{args.output_path}/{args.ice_model}_z{antenna_depth:.3f}.hdf5', mode = 'w') as f:
        f.attrs['antenna_depth'] = antenna_depth
        f.attrs['ice_model'] = args.ice_model
        for key, value in lookup_table['header'].items():
            f.attrs[key] = value

        for key, values in lookup_table[channel_type['name']].items():
            f.create_dataset(name=key, data=values)
