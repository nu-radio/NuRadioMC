"""
Script that benchmarks the speed of the NuRadioMC IO modules

First creates a sample .nur file using the NuRadioMC/test/Veff
example script. Then writes and reads this file n times and reports
the time per event.

"""
from NuRadioReco.modules.io import eventReader, eventWriter
import subprocess
import time
from NuRadioMC import __path__ as nuradiomc_path
import argparse
import os
import logging
logger = logging.getLogger('test-io')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argparser.add_argument(
        '--file', type=str, default=None,
        help='Specify a custom .nur file to use for the benchmark')
    argparser.add_argument(
        '--n', type=int, default=100,
        help='Number of read/write iterations to perform. Default n=100')

    args = argparser.parse_args()
    cleanup_list = [] # keep track of the files we need to clean up afterwards
    try: # we use a try/except structure to always clean up after ourselves
        if args.file is None:
            example_path = os.path.join(nuradiomc_path[0], 'test', 'Veff', '1e18eV')
            cleanup_list += [
                os.path.join(example_path, '1e18_full.hdf5'),
                os.path.join(example_path, 'output.hdf5'),
                os.path.join(example_path, 'output.nur')
            ]

            subprocess.run(['python3', os.path.join(example_path, 'T01generate_event_list.py')])
            subprocess.run([
                'python3', os.path.join(example_path, 'T02RunSimulation.py'),
                '1e18_full.hdf5',
                '../dipole_100m.json', '../config.yaml',
                'output.hdf5', 'output.nur'])
            nur_path = os.path.join(example_path, 'output.nur')
        else:
            nur_path = args.file

        filesize = os.path.getsize(nur_path)

        if args.n >= 1e4:
            logger.error(f'Number of trials {args.n} exceeds 10000, please pick a lower number.')
            exit(1)
        elif args.n * filesize > 2e9:
            logger.error(
                f'Number of trials for chosen file {nur_path} will use more than 2 GB of disk space. '
                'Please reduce number of trials or use a smaller .nur file.')
            exit(2)

        reader = eventReader.eventReader()
        writer = eventWriter.eventWriter()

        # we do the write test first
        reader.begin(nur_path)
        evt_list = list(reader.run())

        if not len(evt_list):
            raise ValueError(f"Found 0 events in nur file {nur_path}, exiting benchmark...")

        os.mkdir('test-output')
        output_files = [os.path.join('test-output', f'test-output-{j:04d}.nur') for j in range(args.n)]
        cleanup_list += output_files

        logger.warning("Starting write benchmark...")
        t0 = time.time()

        for f in output_files:
            writer.begin(f)
            for event in evt_list:
                writer.run(event)
            writer.end()

        dt =  time.time() - t0
        dt_write = dt / (args.n * len(evt_list))

        logger.warning('Starting read benchmark...')
        t0 = time.time()

        for f in output_files:
            reader.begin(f)
            for evt in reader.run():
                pass # TODO - if future formats don't automatically read traces this will impact performance

        dt = time.time() - t0
        dt_read = dt / (args.n * len(evt_list))

        logger.warning(f"Write speed: {dt_write*1e3:-6.2f} ms / event ({args.n * len(evt_list)} events total).")
        logger.warning(f"Read speed : {dt_read*1e3:-6.2f} ms / event ({args.n * len(evt_list)} events total).")

    finally:
        for f in cleanup_list:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass

            # remove test directory
            try:
                os.rmdir('test-output')
            except (OSError, FileNotFoundError):
                pass