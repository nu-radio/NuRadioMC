import h5py

with h5py.File("1e19_n100.hdf5", "r") as f:
    print("Keys in file:")
    for k in f.keys():
        print(" ", k)

with h5py.File("1e19_n100.hdf5", "r") as f:
    zenith = f["zeniths"][:]
    azimuth = f["azimuths"][:]
    energies = f["energies"][:]

print("Zenith range:", zenith.min(), zenith.max(),len(zenith))
print("Azimuth range:", azimuth.min(), azimuth.max())
print("energies range:", zenith.min(), zenith.max(),len(energies))

with h5py.File("1e19_n100.hdf5", "r") as f:
    egid = f["event_group_ids"][:]

print("Number of primaries:", len(set(egid)))
print("Number of showers:", len(egid))





import h5py
import numpy as np

with h5py.File("1e19_n100.hdf5", "r") as f:
    zeniths = f["zeniths"][:]
    egid = f["event_group_ids"][:]

# get unique neutrino IDs and first occurrence
unique_ids, first_idx = np.unique(egid, return_index=True)

primary_zeniths = zeniths[first_idx]

print("Number of primaries:", len(primary_zeniths))
print("Zenith range:", primary_zeniths.min(), primary_zeniths.max())