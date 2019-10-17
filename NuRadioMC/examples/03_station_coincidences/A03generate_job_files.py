from NuRadioMC.EvtGen import generator
from NuRadioReco.utilities import units
import numpy as np
import glob
import os
import sys


"""
This example creates job files for simulations to check for coincidences.

"""

n_events_per_file = 300

try:
    software = sys.argv[0]
    # specify the path to the software directory (where NuRadioMC, NuRadioReco and radiotools are installed in)
    working_dir   = sys.argv[1]
    #where you want the simulation to go
    conf_dir   = sys.argv[2]  # where are the config files
    presim_dir = sys.argv[3]
    # Directory with simulations ()

except:
    print("Usage python A02generate_job_files.py software working_dir presim_dir")
    print("Using default values")
    conf_dir = "/pub/arianna/NuRadioMC/examples/03_station_coincidences/"
    working_dir = conf_dir
    software = '/data/users/jcglaser/software'
    presim_dir = os.path.join(working_dir, 'presim/output')

config_file = os.path.join(working_dir, 'config.yaml')


if not os.path.exists(os.path.join(working_dir, "output")):
    os.makedirs(os.path.join(working_dir, "output"))
if not os.path.exists(os.path.join(working_dir, "run")):
    os.makedirs(os.path.join(working_dir, "run"))
if not os.path.exists(os.path.join(working_dir, "input")):
    os.makedirs(os.path.join(working_dir, "input"))

for iI, filename in enumerate(sorted(glob.glob(os.path.join(presim_dir,"*.hdf5")))):
    current_folder = os.path.splitext(os.path.basename(filename))[0]
    t1 = os.path.join(working_dir, "output", current_folder)
    if(not os.path.exists(t1)):
        os.makedirs(t1)
    t1 = os.path.join(working_dir, 'run', current_folder)
    if(not os.path.exists(t1)):
        os.makedirs(t1)
    t1 = os.path.join(working_dir, 'input', current_folder)
    if(not os.path.exists(t1)):
        os.makedirs(t1)

    output_filename = os.path.join(working_dir, 'input', current_folder, os.path.basename(filename))

    print('saving files to {}'.format(output_filename))

    generator.split_hdf5_input_file(filename, output_filename, n_events_per_file)

    for iF, filename2 in enumerate(sorted(glob.glob(os.path.join(working_dir, 'input', current_folder, "{}*".format(os.path.basename(output_filename[:-1])))))):
        print('generating job submission files for {}'.format(filename2))
        detector_file = os.path.join(working_dir, 'horizontal_spacing_detector.json')
        output_filename = os.path.join("output", current_folder, os.path.basename(filename2))
        cmd = "python {} {} {} {} {}\n".format(os.path.join(working_dir, 'E06RunSimPreprocess.py'), filename2,
                                               detector_file, config_file, output_filename)

        header = '#!/bin/bash\n'
        header += '#$ -N HS_{}_{}\n'.format(iI, iF)
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

        with open(os.path.join('run', current_folder, os.path.basename(filename2) + ".sh"), 'w') as fout:
            fout.write(header)
            fout.write(cmd)
