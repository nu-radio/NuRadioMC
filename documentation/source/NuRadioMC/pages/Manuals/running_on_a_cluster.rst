Running on a cluster
====================
NuRadioMC comes with the tools that simplifies running simulations on many cores in parallel on clusters. On this page, we explain the necessary steps using the HPC cluster of UC Irvine as example. 

1. Generate input files
-----------------------

The event generator module has the feature to split up the data set into several smaller files. A good number of events per batch is 10,000 - 100,000 which takes a couple of hours to simulate. To simulate the sensitivity of a detector, we need 1M - 10M events per energy, hence, we end up with ~100 jobs per energy that we can all run in parallel. The following example code shows how to generate 1 million events in batches of 10,000 events per file. 

.. code-block:: Python

    from NuRadioMC.utilities import units
    from NuRadioMC.EvtGen.generator import generate_eventlist_cylinder

    # generate one event list at 1e18 eV with 1M neutrinos with 10k events per file
    generate_eventlist_cylinder('/pub/arianna/NuRadioMC/input/1e18/1e18_n1e6.hdf5',
                                n_events=1e6, n_events_per_file=1e4,
                                Emin=1e18 * units.eV, Emax=1e18 * units.eV, fiducial_rmin=0,
                                fiducial_rmax=5 * units.km, fiducial_zmin=-2.7 * units.km, fiducial_zmax=0)

This script will create 100 hdf5 input files with filenames ``1e18_n1e6.hdf5.part0001`` to ``1e18_n1e6.hdf5.part0100``.

2. Generate job \*.sh scripts
-----------------------------

Most job schedulers require a bash script that sets the environment and executes the software. In the following script, the details of the NuRadioMC simulation are specified (python steering script, config, detector description, ...) and for each input file on job script is created. 

.. code-block:: Python

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
    input_dir = "/pub/arianna/NuRadioMC/input/"
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

3. Submit jobs to the cluster
-------------------------------

In case of the grid engine scheduler, all job files can be submitted with this bash line

.. code-block:: bash

    for f in $(ls *.sh); do qsub $f; done;


4. Merge individual hdf5 output files
-------------------------------------

It is often more convenient to work with a single output file (per energy). Each individual hdf5 file is typically small, hence, merging the simulation result back into a single file is convenient. NuRadioMC comes with the tools to do that. One thing that needs special consideration is that by default only triggered events are saved in the output file. To be able to calculate the effective volume, we need to keep track of the total number of simulated events, which is stored in the attribute 'n_events'. Therefore, third-party merging tools can't be used out of the box. The NuRadioMC merging tool automatically calculates the sum from all individual files, so that the merged file contains the correct total event count. 


To merge all files execute

.. code-block:: bash

    python ../NuRadioMC/utilities/merge_hdf5.py /path/to/my/output/files/


