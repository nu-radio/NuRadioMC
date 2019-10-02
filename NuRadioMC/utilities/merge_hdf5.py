import glob
import os
import sys
import numpy as np
from collections import OrderedDict
import h5py
import argparse
import logging
logger = logging.getLogger("HDF5-merger")
logging.basicConfig(level=logging.DEBUG)
logger.setLevel(logging.WARNING)


def merge2(filenames, output_filename):
    data = OrderedDict()
    attrs = OrderedDict()
    groups = OrderedDict()
    group_attrs = OrderedDict()
    n_data = {}
    n_groups = {}
    non_empty_filenames = []
    n_events_total = 0

    for f in filenames:
        logger.info("adding file {}".format(f))
        fin = h5py.File(f, 'r')
        n_events_total += fin.attrs['n_events']
        logger.debug(f"increasing total number of events by {fin.attrs['n_events']:d} to {n_events_total:d} ")

        if(np.sum(np.array(fin['triggered'])) == 0):
            logger.info(f"file {f} contains no events")
        else:
            non_empty_filenames.append(f)
            logger.debug(f"file {f} contains {np.sum(np.array(fin['triggered']))} triggered events.")

        data[f] = {}
        groups[f] = {}

        for key in fin:
            if isinstance(fin[key], h5py._hl.group.Group):
                groups[f][key] = {}
                if(key not in n_groups):
                    n_groups[key] = {}
                for key2 in fin[key]:
                    groups[f][key][key2] = fin[key][key2][...]
                    if(key2 not in n_groups[key]):
                        n_groups[key][key2] = 0
                    n_groups[key][key2] += len(groups[f][key][key2])
                if(key not in group_attrs):
                    group_attrs[key] = {}
                    for key2 in fin[key].attrs:
                        group_attrs[key][key2] = fin[key].attrs[key2]
                else:
                    for key2 in fin[key].attrs:
                        if(not np.all(group_attrs[key][key2] == fin[key].attrs[key2])):
                            logger.warning(f"attribute {key2} of group {key} of file {filenames[0]} and {f} are different ({group_attrs[key][key2]} vs. {fin[key].attrs[key2]}. Using attribute value of first file, but you have been warned!")
            else:
                data[f][key] = fin[key][...]
                if(key not in n_data):
                    n_data[key] = 0
                n_data[key] += len(data[f][key])

        for key in fin.attrs:
            if(key not in attrs):
                attrs[key] = fin.attrs[key]
            else:
                if(key != 'trigger_names'):
                    if(not np.all(attrs[key] == fin.attrs[key])):
                        if(key == "n_events"):
                            logger.warning(f"number of events in file {filenames[0]} and {f} are different ({attrs[key]} vs. {fin.attrs[key]}. We keep track of the total number of events, but in case the simulation was performed with different settings per file (e.g. different zenith angle bins), the averaging might be effected.")
                        else:
                            logger.warning(f"attribute {key} of file {filenames[0]} and {f} are different ({attrs[key]} vs. {fin.attrs[key]}. Using attribute value of first file, but you have been warned!")
            if((('trigger_names' not in attrs) or (len(attrs['trigger_names']) == 0)) and 'trigger_names' in fin.attrs):
                attrs['trigger_names'] = fin.attrs['trigger_names']
        fin.close()

    # create data sets
    logger.info("creating data sets")
    fout = h5py.File(output_filename, 'w')
    if(len(non_empty_filenames)):
        keys = data[non_empty_filenames[0]]
        for key in keys:
            logger.info(f"merging key {key}")
            all_files_have_key = True
            for f in non_empty_filenames:
                if(not key in data[f]):
                    logger.debug(f"key {key} not in {f}")
                    all_files_have_key = False
            if(not all_files_have_key):
                logger.warning(f"not all files have the key {key}. This key will not be present in the merged file.")
                continue
            shape = list(data[non_empty_filenames[0]][key].shape)
            shape[0] = n_data[key]

            tmp = np.zeros(shape, dtype=data[non_empty_filenames[0]][key].dtype)

            i = 0
            for f in non_empty_filenames:
                tmp[i:(i + len(data[f][key]))] = data[f][key]
                i += len(data[f][key])

            fout.create_dataset(key, tmp.shape, dtype=tmp.dtype,
                                compression='gzip')[...] = tmp

        keys = groups[non_empty_filenames[0]]
        for key in keys:
            logger.info("writing group {}".format(key))
            g = fout.create_group(key)
            for key2 in groups[non_empty_filenames[0]][key]:
                logger.info("writing data set {}".format(key2))
                all_files_have_key = True
                for f in non_empty_filenames:
                    if(not key2 in groups[f][key]):
                        logger.debug(f"key {key2} of group {key} not in {f}")
                        all_files_have_key = False
                if(not all_files_have_key):
                    logger.warning(f"not all files have the key {key2}. This key will not be present in the merged file.")
                    continue

                shape = list(groups[non_empty_filenames[0]][key][key2].shape)
                shape[0] = n_groups[key][key2]

                tmp = np.zeros(shape, dtype=groups[non_empty_filenames[0]][key][key2].dtype)
                i = 0
                for f in non_empty_filenames:
                    tmp[i:(i + len(groups[f][key][key2]))] = groups[f][key][key2]
                    i += len(groups[f][key][key2])

                g.create_dataset(key2, shape, dtype=groups[non_empty_filenames[0]][key][key2].dtype,
                                 compression='gzip')[...] = tmp
            # save group attributes
            for key2 in group_attrs[key]:
                fout[key].attrs[key2] = group_attrs[key][key2]
        # save all atrributes
        attrs['n_events'] = n_events_total
        for key in attrs:
            fout.attrs[key] = attrs[key]
    else:  # now handle the case
        logger.warning("All files are empty. Copying content of first file to output file and keepting track of total number of simulated events.")
        # all files are empty, so just copy the content of the first file (attributes and empyt data sets) to the output file
        # update n_events attribute with the total number of events
        fin = h5py.File(filenames[0], 'r')
        for key in fin.attrs:
            if(key == "n_events"):
                fout.attrs[key] = n_events_total
            else:
                fout.attrs[key] = fin.attrs[key]
        for key in fin:
            if isinstance(fin[key], h5py._hl.group.Group):
                g = fout.create_group(key)
                for key2 in fin[key]:
                    g.create_dataset(key2, fin[key][key2].shape, dtype=fin[key][key2].dtype,
                                     compression='gzip')[...] = fin[key][key2]
                for key2 in fin[key].attrs:
                    g.attrs[key2] = fin[key].attrs[key2]
            else:
                fout.create_dataset(key, fin[key].shape, dtype=fin[key].dtype,
                                    compression='gzip')[...] = fin[key]

#     # save all data to hdf5
#     for key in data[filenames[0]]:
#         print("writing data set {}".format(key))
#         i = 0
#         for f in data:
#             fout[key][i:(i+len(data[f][key]))] = data[f][key]
#             i += len(data[f][key])
    # save all group data to hdf5
#     for key in groups[filenames[0]]:
#         print("writing group {}".format(key))
#         for key2 in groups[filenames[0]][key]:
#             print("writing data set {}".format(key2))
#             i = 0
#             for f in groups:
#                 fout[key][key2][i:(i+len(groups[f][key][key2]))] = groups[f][key][key2]
#                 i += len(groups[f][key][key2])
#         # save group attributes
#         for key2 in group_attrs[key]:
#             fout[key].attrs[key2] = group_attrs[key][key2]
#

    fout.close()


if __name__ == "__main__":
    """
    merges multiple hdf5 output files into one single files.
    The merger module automatically keeps track of the total number
    of simulated events (which are needed to correctly calculate the effective volume).

    The script expects that the folder structure is
    ../output/energy/*.hdf5.part????

    Optional log level setting to either set DEBUG, INFO, or WARNING to the readout. Example: add --loglevel DEBUG when calling script to set loglevel to DEBUG. 
    """
    parser = argparse.ArgumentParser(description='Merge hdf5 files')
    parser.add_argument('files', nargs='+', help='input file or files')
    parser.add_argument('--loglevel', metavar='level', help='loglevel set to either DEBUG, INFO, or WARNING')
    args = parser.parse_args()

    if args.loglevel is not None:
        log_val = eval(f'logging.{args.loglevel}')
        logger.setLevel(log_val)

    if(len(args.files) < 1):
        print("usage: python merge_hdf5.py /path/to/simulation/output/folder\nor python merge_hdf5.py outputfilename input1 input2 ...")
    elif(len(args.files) == 1):
        filenames = glob.glob("{}/*/*.hdf5.part????".format(args.files[0]))
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
                output_filename = os.path.join(a, d[1])  # remove subfolder from filename
                if(os.path.exists(output_filename)):
                    logger.error('file {} already exists, skipping'.format(output_filename))
                else:
                    #                 try:
                    input_files = np.array(sorted(glob.glob(filename + '.part????')))
                    mask = np.array([os.path.getsize(x) > 1000 for x in input_files], dtype=np.bool)
                    if(np.sum(~mask)):
                        logger.warning("{:d} files were deselected because their filesize was to small".format(np.sum(~mask)))

                    merge2(input_files[mask], output_filename)
    #                 except:
    #                     print("failed to merge {}".format(filename))
    elif(len(args.files) > 1):
        output_filename = args.files[0]
        if(os.path.exists(output_filename)):
            logger.error('file {} already exists, skipping'.format(output_filename))
        else:
            input_files = args.files[1:]
            merge2(input_files, output_filename)
