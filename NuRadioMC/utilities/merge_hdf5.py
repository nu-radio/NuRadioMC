import glob
import os
import sys
import numpy as np
from NuRadioMC.utilities.hdf5_manipulator import merge

"""
merges multiple hdf5 output files into one single files.
The merger module automatically keeps track of the total number
of simulated events (which are needed to correctly calculate the effective volume).

The script expects that the folder structure is
../output/energy/*.hdf5.part????
"""
if(len(sys.argv) != 2):
    print("usage: python merge_hdf5.py /path/to/simulation/output/folder")
else:
    filenames = glob.glob("{}/*/*.hdf5.part????".format(sys.argv[1]))
    filenames2 = []
    for i, filename in enumerate(filenames):
        filename, ext = os.path.splitext(filename)
        if(ext != '.hdf5'):
            if(filename not in filenames2):
                d = os.path.split(filename)
                a, b = os.path.split(d[0])
                filenames2.append(filename)

    for filename in filenames2:
        if(os.path.splitext(filename)[1] == '.hdf5'):
            d = os.path.split(filename)
            a, b = os.path.split(d[0])
            output_filename = os.path.join(a, d[1])  #remove subfolder from filename
            if(os.path.exists(output_filename)):
                print('file {} already exists, skipping'.format(output_filename))
            else:
#                 try:
                    input_files = np.array(sorted(glob.glob(filename + '.part????')))
                    mask = np.array([os.path.getsize(x) > 1000 for x in input_files], dtype=np.bool)
                    if(np.sum(~mask)):
                        print("{:d} files were deselected because their filesize was to small".format(np.sum(~mask)))


                    merge.merge_data_filenames(input_files[mask], output_filename)
#                 except:
#                     print("failed to merge {}".format(filename))
