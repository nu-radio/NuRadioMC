import h5py
import numpy as np
import argparse
from radiotools import helper as hp
from NuRadioReco.utilities import units

keys_event = [
 u'event_group_ids',
 u'azimuths',
 u'energies',
 u'flavors',
 u'inelasticity',
 u'interaction_type',
 u'multiple_triggers',
 u'n_interaction',
 u'triggered',
 u'xx',
 u'yy',
 u'zeniths',
 u'multiple_triggers',
 u'zz',
 u'weights'
 ]

station_keys = [
    u'max_amp_shower_and_ray',
     u'ray_tracing_C0',
     u'ray_tracing_C1',
     u'ray_tracing_solution_type',
     u'travel_times',
     u'travel_distances',
     ]

station_keys_3dim = [
 u'launch_vectors',
 u'polarization',
 u'receive_vectors',
 ]

station_keys_event = [
     u'maximum_amplitudes',
 u'maximum_amplitudes_envelope']


def dump(filename):
    print(f"!!!!!!!!!!!! dumping relevant file content of {filename}:!!!!!!!!!!!!!!!")
    fin = h5py.File(filename, 'r')
    stations = []
    for key in fin.keys():
        if(key.startswith("station_")):
            stations.append(key)
    event_group_ids = np.array(fin['event_group_ids'])
    for iE, evt_gid in enumerate(event_group_ids):
        t = "index, "
        for key in keys_event:
            t += f"{key}, "
        print(t)
        t = f"{iE} "
        for key in keys_event:
            t += f"{fin[key][iE]} "
        print(t)
        t = "stationid, channelid, rayid, "
        for key in station_keys:
            t += f"{key}, "
        for key in station_keys_3dim:
            t += f"{key}, "
        t += f" zen, az"
        print(t)
        for station in stations:
            if not 'ray_tracing_C0' in station:
                print(f'{station} has not entries')
                continue

            nCh, nR = np.array(fin[station]['ray_tracing_C0'][iE]).shape
            for iCh in range(nCh):
                for iR in range(nR):
                    t = f"\t{station} {iCh} {iR}: "
                    for key in station_keys:
                        t += f"{fin[station][key][iE][iCh][iR]:.9g} "
                    for key in station_keys_3dim:
                        t += "("
                        for iD in range(3):
                            t += f"{fin[station][key][iE][iCh][iR][iD]:.5g},"
                        t += ") "
                    zen, az = hp.cartesian_to_spherical(*np.array(fin[station]["receive_vectors"][iE][iCh][iR]))
                    t += f" {zen/units.deg:.2f} {az/units.deg:.2f}"
                    print(t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge hdf5 files')
    parser.add_argument('file', help='input file or files')
    args = parser.parse_args()
    dump(args.file)
