from NuRadioMC.EvtGen import generator
from NuRadioReco.utilities import units
import numpy as np
import glob
import os
import sys


"""
This example creates job files for a pre simulation that saves all events that trigger a central station

"""

n_events_per_file = 300

try:
    software = sys.argv[0]
    # specify the path to the software directory (where NuRadioMC, NuRadioReco and radiotools are installed in)
    working_dir   = sys.argv[1]
    #where you want the simulation to go
    conf_dir   = sys.argv[2]  # where are the config files
    input_dir = sys.argv[3]
    # Directory with simulations ()

except:
    print("Usage python A02generate_job_files_presim.py software working_dir conf_dir input_dir")
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
