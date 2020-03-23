import os
import argparse
import logging
import math
from six import iteritems
import numpy as np
import h5py
logger = logging.getLogger("HDF5-split")
logging.basicConfig(level=logging.DEBUG)
logger.setLevel(logging.WARNING)


def split_hdf5_input_file(input_filename, output_filename, number_of_events_per_file):
    """
    splits up an existing hdf5 file into multiple subfiles

    Parameters
    ----------
    input_filename: string
        the input filename
    output_filename: string
        the desired output filename (if multiple files are generated, a 'part000x' is appended to the filename
    n_events_per_file: int (optional, default None)
        the number of events per file
    """
    fin = h5py.File(input_filename, 'r')
    attributes = {}
    data = {}
    groups = {}
    n_groups = {}
    n_data = {}
    group_attrs = {}

    for key in fin:
        if isinstance(fin[key], h5py._hl.group.Group):
            groups[key] = {}
            if(key not in n_groups):
                n_groups[key] = {}
            for key2 in fin[key]:
                groups[key][key2] = fin[key][key2][...]
                if(key2 not in n_groups[key]):
                    n_groups[key][key2] = 0
                n_groups[key][key2] += len(groups[key][key2])
            if(key not in group_attrs):
                group_attrs[key] = {}
                for key2 in fin[key].attrs:
                    group_attrs[key][key2] = fin[key].attrs[key2]
            else:
                for key2 in fin[key].attrs:
                    if(not np.all(group_attrs[key][key2] == fin[key].attrs[key2])):
                        logger.warning(f"attribute {key2} of group {key} of file {input_filename} are different ({group_attrs[key][key2]} vs. {fin[key].attrs[key2]}. Using attribute value of first file, but you have been warned!")
        else:
            data[key] = fin[key][...]
            if(key not in n_data):
                n_data[key] = 0
            n_data[key] += len(data[key])

    for key, value in iteritems(fin.attrs):
        attributes[key] = value
#     logger.info(f"setting number of events from {attributes['n_events']} to the actual number of events in this file {len(data_sets['event_ids'])}")
    fin.close()

    n_events = len(data['event_ids'])
    logger.info("saving {} events in total".format(n_events))
    total_number_of_events = attributes['n_events']

    n_files = math.ceil(n_events / number_of_events_per_file)
    for iFile in range(n_files):
        filename2 = output_filename + ".part{:04}".format(iFile)
        logger.debug(f"saving file {iFile} with {number_of_events_per_file} events to {filename2}")
        fout = h5py.File(filename2, 'w')
        for key, value in attributes.items():
            fout.attrs[key] = value
        fout.attrs['total_number_of_events'] = total_number_of_events / n_files
        fout.attrs['n_events'] = number_of_events_per_file

        for key, value in data.items():
            if value.dtype.kind == 'U':
                fout[key] = [np.char.encode(c, 'utf8') for c in value[iFile * number_of_events_per_file:(iFile + 1) * number_of_events_per_file]]
            else:
                fout[key] = value[iFile * number_of_events_per_file:(iFile + 1) * number_of_events_per_file]

        i1, i2 = iFile * number_of_events_per_file, (iFile + 1) * number_of_events_per_file
        for key in groups:
            logger.info("writing group {}".format(key))
            g = fout.create_group(key)
            for key2 in groups[key]:
                logger.info("writing data set {}".format(key2))

                shape = list(groups[key][key2][i1:i2].shape)
#                 shape[0] = n_groups[key][key2]

                tmp = groups[key][key2][i1:i2]

                g.create_dataset(key2, shape, dtype=groups[key][key2].dtype,
                                 compression='gzip')[...] = tmp
            # save group attributes
            for key2 in group_attrs[key]:
                fout[key].attrs[key2] = group_attrs[key][key2]

        fout.close()


if __name__ == "__main__":
    """
    Splits up a hdf5 file into multiple files
    
    Parameters
    -----------
    file: str
        the input file
    outputfolder: str
        the ouput folder
    n_events: int
        the maximum number of events in each file. The last file will contain less events. 
    Optional log level setting to either set DEBUG, INFO, or WARNING to the readout. Example: add --loglevel DEBUG when calling script to set loglevel to DEBUG. 
    """
    parser = argparse.ArgumentParser(description='Merge hdf5 files')
    parser.add_argument('file', type=str, help='input file')
    parser.add_argument('outputfolder', type=str, help='output folder')
    parser.add_argument('n_events', type=int, help='number of events per file')
    parser.add_argument('--loglevel', metavar='level', help='loglevel set to either DEBUG, INFO, or WARNING')
    args = parser.parse_args()

    if args.loglevel is not None:
        log_val = eval(f'logging.{args.loglevel}')
        logger.setLevel(log_val)

    if(not os.path.exists(args.outputfolder)):
        os.makedirs(args.outputfolder)
    input_filename = os.path.basename(args.file)

    split_hdf5_input_file(args.file, os.path.join(args.outputfolder, input_filename), args.n_events)
