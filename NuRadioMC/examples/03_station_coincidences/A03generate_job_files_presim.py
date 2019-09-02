from NuRadioMC.EvtGen import generator
from NuRadioMC.utilities import units
import numpy as np
import glob
import os
import sys


"""
This example creates job files for a pre simulation that saves all events that trigger a central station

This example creates job files for simulations to check for coincidences. The
number of events per file is controlled with n_events_per_file.

The arguments to pass to this file are, in order:

software: the directory where NuRadioMC, NuRadioReco and radiotools are installed

working_dir: the directory where the job files are going to be stored (input,
run and output files)

conf_dir: the directory where the configuration file is located

input_dir: the directory where the output files from the presimulation of the
central station can be found. They are used as input files to save computing time.

The default parameters are those used for Christian's cluster, but they can be
changed below.

*** IMPORTANT ***

This file is to be used only if the central station has been presimulated.
If central station has not been presimulated, one should use
A03generate_job_files.py.

Do not forget to change the header variable to include all the instructions
and paths that are needed to operate your cluster. Otherwise the example will
not work.

"""

n_events_per_file = 300

try:
    software = sys.argv[1]
    # specify the path to the software directory (where NuRadioMC, NuRadioReco and radiotools are installed in)
    working_dir   = sys.argv[2]
    #where you want the simulation to go
    conf_dir   = sys.argv[3]  # where are the config files
    input_dir = sys.argv[4]
    # Directory with simulations ()

except:
    print("Usage python A03generate_job_files_presim.py software working_dir conf_dir input_dir")
    print("Using default values")
    conf_dir = "/pub/arianna/NuRadioMC/examples/03_station_coincidences/"
    working_dir = os.path.join(conf_dir, "presim")
    software = '/data/users/jcglaser/software'
    input_dir = '/pub/arianna/NuRadioMC/input_1e4/'

config_file = os.path.join(conf_dir, 'config.yaml')


if not os.path.exists(os.path.join(working_dir, "output")):
    os.makedirs(os.path.join(working_dir, "output"))
if not os.path.exists(os.path.join(working_dir, "run")):
    os.makedirs(os.path.join(working_dir, "run"))

for iI, filename in enumerate(sorted(glob.glob(os.path.join(input_dir, "*/*.hdf*")))):
    current_folder = os.path.split(os.path.dirname(filename))[-1]
    t1 = os.path.join(working_dir, "output", current_folder)
    if(not os.path.exists(t1)):
        os.makedirs(t1)
    t1 = os.path.join(working_dir, 'run', current_folder)
    if(not os.path.exists(t1)):
        os.makedirs(t1)

#     print('generating job submission files for {}'.format(filename))
    detector_file = os.path.join(conf_dir, 'single_position.json')
    output_filename = os.path.join(working_dir, "output", current_folder, os.path.basename(filename))
    cmd = "python {} {} {} {} {}\n".format(os.path.join(conf_dir, 'E06RunSimPreprocess.py'),
                                           filename, detector_file, config_file, output_filename)

    header = '#!/bin/bash\n'
    header += '#$ -N PreSim_{}\n'.format(iI)
    header += '#$ -j y\n'
    header += '#$ -V\n'
    header += '#$ -q grb,grb64\n'
#         header += '#$ -m \n'
    header += '#$ -o {}\n'.format(os.path.join(working_dir, 'run'))
    # add the software to the PYTHONPATH
    header += 'export PYTHONPATH={}/NuRadioMC:$PYTHONPATH\n'.format(software)
    header += 'export PYTHONPATH={}/NuRadioReco:$PYTHONPATH \n'.format(software)
    header += 'export PYTHONPATH={}/radiotools:$PYTHONPATH \n'.format(software)
    header += 'cd {} \n'.format(working_dir)

    with open(os.path.join(working_dir, 'run', current_folder, os.path.basename(filename) + ".sh"), 'w') as fout:
        fout.write(header)
        fout.write(cmd)
