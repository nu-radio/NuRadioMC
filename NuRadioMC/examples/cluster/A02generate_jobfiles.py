import glob
import os

# define the base directory of this job 
base_dir ="/pub/arianna/NuRadioMC/station_designs"
# specify the NuRadioMC python steering file (needs to be in base_dir)  
detector_sim = 'T02RunSimulation.py'
# specify detector description (needs to be in base_dir)
detector_filename = 'surface_station_1GHz.json'
# specify directory that contains the detector descriptions
det_dir =  os.path.join(base_dir, "detectors")
# specify the NuRadioMC config file that should be used for this simulation
config_file =  os.path.join(base_dir, "config_5.yaml")
# specify a working directory for this specific simulation run
working_dir =  os.path.join(base_dir, "suface_station_1GHz/05")
# specify the directory containing the input event files, this directory needs to contain separate folders
input_dir = "/pub/arianna/NuRadioMC/input_8km_1e7_1e5/"
# specify the path to the software directory (where NuRadioMC, NuRadioReco and radiotools are installed in)
software = '/data/users/jcglaser/software'

# run and output directories are created automatically if not yet present
if not os.path.exists(os.path.join(working_dir, "output")):
    os.makedirs(os.path.join(working_dir, "output"))
if not os.path.exists(os.path.join(working_dir, "run")):
    os.makedirs(os.path.join(working_dir, "run"))

# loop over all input event files and create a job script for each input file. 
for iF, filename in enumerate(sorted(glob.glob(os.path.join(input_dir, '*/*.hdf5.*')))):
    current_folder = os.path.split(os.path.dirname(filename))[-1]
    detector_file = os.path.join(det_dir, detector_filename)
    # check if subfolder for energies exist
    t1 = os.path.join(working_dir, "output", current_folder)
    if(not os.path.exists(t1)):
        os.makedirs(t1)
    t1 = os.path.join(working_dir, 'run', current_folder)
    if(not os.path.exists(t1)):
        os.makedirs(t1)
    output_filename = os.path.join(working_dir, "output", current_folder, os.path.basename(filename))
    cmd = "python {} {} {} {} {}\n".format(os.path.join(base_dir, detector_sim), filename, detector_file, config_file,
                                           output_filename)

    # here we add specific settings for the grid engine job scheduler, this part need to be adjusted to the specifics 
    # of your cluster
    header = '#!/bin/bash\n'
    header += '#$ -N C_{}\n'.format(iF)
    header += '#$ -j y\n'
    header += '#$ -V\n'
    header += '#$ -q grb,grb64\n'
    header += '#$ -ckpt restart\n'  # restart jobs in case of a node crash
    header += '#$ -o {}\n'.format(os.path.join(working_dir, 'run'))
    
    # add the software to the PYTHONPATH
    header += 'export PYTHONPATH={}/NuRadioMC:$PYTHONPATH\n'.format(software)
    header += 'export PYTHONPATH={}/NuRadioReco:$PYTHONPATH \n'.format(software)
    header += 'export PYTHONPATH={}/radiotools:$PYTHONPATH \n'.format(software)
    header += 'cd {} \n'.format(working_dir)

    with open(os.path.join(working_dir, 'run', current_folder, os.path.basename(filename) + ".sh"), 'w') as fout:
        fout.write(header)
        fout.write(cmd)
