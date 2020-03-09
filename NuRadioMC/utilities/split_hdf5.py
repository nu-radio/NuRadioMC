from NuRadioMC.EvtGen.generator import split_hdf5_input_file
import os
import argparse
import logging
logger = logging.getLogger("HDF5-split")
logging.basicConfig(level=logging.DEBUG)
logger.setLevel(logging.WARNING)

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

    split_hdf5_input_file(args.file, os.path.join(args.outputfolder, args.file), args.n_events)
