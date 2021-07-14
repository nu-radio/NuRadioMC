import h5py
import numpy as np

f = h5py.File('1e19_n1e3_outp_ARA.hdf5', 'r')
f2 = h5py.File('dummy_inp.hdf5', 'w')

for i in f.keys():
    #print(i, f[i][0])
    if i == 'interaction_type' or i == 'shower_type':
        f2.create_dataset(i, (1,), dtype=h5py.string_dtype(encoding='ascii'))
        f2[i][0] = f[i][0]
    elif i != 'station_101' and i != 'multiple_triggers' and i != 'triggered' and i != 'shower_realization_Alvarez2009':
        f2.create_dataset(i, (1,))#, dtype=type(f[i][0]))
        f2[i][0] = f[i][0]
    else:
        pass 

f.close()
f2.close()
