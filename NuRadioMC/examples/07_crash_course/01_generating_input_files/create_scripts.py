"""
This is an example on how to create automated jobs for the DESY cluster.
This file takes all the HDF5 input files in folder that contain 'rad' on
the name and creates jobs. Be sure to change all the local path variables
and the files you want to use for the simulation. See comments below.

The files should be organised in the way the T01*py scripts on this folder do it.
"""

import glob
import os

# Define the base directory of this job. Be sure to change this
base_dir ="/lustre/fs22/group/radio/garcia/simulations/my_simulation"
# specify the NuRadioMC python steering file (needs to be in base_dir)
detector_sim = 'T02AliasGreenland.py'
# specify detector description
detector_filename = 'RNO_phased_100m_0.5GHz.json'
# specify the NuRadioMC config file that should be used for this simulation
config_file =  os.path.join(base_dir, "config.yaml")
# specify the path to the software directory (where NuRadioMC, NuRadioReco and radiotools are installed in)
# Change it to your directory!
software = '/afs/ifh.de/group/radio/scratch/garcia/software/'



thetamode = True
nur_file = False

folders = [ folder for folder in os.listdir('.') if os.path.isdir(folder) ]

"""
You can change the string pattern of your input folders if you use a different one.
"""
if (thetamode == True):
	folders = [ folder for folder in folders if folder.find('rad') != -1 ]

for folder in folders:

	if not thetamode:
		inputfolders = ['']
	else:
		inputfolders = [ subfolder for subfolder in os.listdir(folder)
						if os.path.isdir(os.path.join(folder,subfolder)) and subfolder.find('output') == -1 ]


	for inputfolder in inputfolders:

		# specify directory that contains the detector descriptions
		det_dir =  base_dir
		# specify a working directory for this specific simulation run
		working_dir =  os.path.join(base_dir, folder, inputfolder)
		output_dir = os.path.join(base_dir, folder, "output")
		# specify the directory containing the input event files, this directory needs to contain separate folders
		input_dir = os.path.join(base_dir, folder, inputfolder, 'input')

		# run and output directories are created automatically if not yet present
		if not os.path.exists(output_dir):
		    os.makedirs(output_dir)
		if not os.path.exists(os.path.join(working_dir, "run")):
		    os.makedirs(os.path.join(working_dir, "run"))

		# loop over all input event files and create a job script for each input file.
		for iF, filename in enumerate(sorted(glob.glob(os.path.join(input_dir, '*.hdf5*')))):
			detector_file = os.path.join(det_dir, detector_filename)
			# check if subfolder for energies exist
			#t1 = os.path.join(working_dir, "output")
			#if(not os.path.exists(t1)):
			#os.makedirs(t1)
			t1 = os.path.join(working_dir, 'run')
			if(not os.path.exists(t1)):
				os.makedirs(t1)
			output_filename = os.path.join(output_dir, 'output'+os.path.basename(filename))
			copy  = "cp {} .\n".format(os.path.join(base_dir, detector_sim))
			copy += "cp {} .\n".format(filename)
			copy += "cp {} .\n".format(detector_file)
			copy += "cp {} .\n".format(config_file)

			output_nur = output_filename + '.nur'
			cmd  = "/afs/ifh.de/group/radio/software/anaconda_p3/bin/python3.7 "
			cmd += "{} ".format(detector_sim)
			cmd += "--inputfilename {} ".format(os.path.basename(filename))
			cmd += "--detectordescription {} ".format(os.path.basename(detector_file))
			cmd += "--config {} ".format(os.path.basename(config_file))
			cmd += "--outputfilename {} ".format(output_filename)
			if nur_file:
				output_nur = output_filename + '.nur'
				cmd += "--outputfilenameNuRadioReco {} ".format(output_nur)
			cmd += '\n'
			copy_back = ''
			copy_back += "cp {} {} \n".format(os.path.basename(output_filename), output_filename)

			# here we add specific settings for the grid engine job scheduler,
			# this part need to be adjusted to the specifics
			# of your cluster
			header = '#!/bin/zsh\n'
			header += '#$ -S /bin/zsh\n'
			# Maximum CPU time
			header += '#$ -l h_cpu=48:00:00\n'
			# Maximum memory allocated
			header += '#$ -l h_rss=5G\n'
			# Maximum
			header += '#$ -l tmpdir_size=1G\n'
			#(stderr and stdout are merged together to stdout)
			header += '#$ -j y\n'
			### send mail if the process aborts. Remove two of the hashtags to enable
			header += '###$ -m a\n'
			# execute job from current directory and not relative to your home directory
			header += '##$ -cwd\n'
			# send output files to the trash, COMMENT TO SEE THE OUTPUT AND DEBUG
			header += '#$ -o {}\n'.format(os.path.join(working_dir, 'run'))
			header += 'hostname\n'
			header += 'echo $TMPDIR\n'
			header += 'df -h\n'
			header += '/usr/sbin/xfs_quota -c "quota -u garcia"\n'

			# add the software to the PYTHONPATH
			header += 'export PYTHONPATH={}/NuRadioMC:$PYTHONPATH\n'.format(software)
			header += 'export PYTHONPATH={}/NuRadioReco:$PYTHONPATH \n'.format(software)
			header += 'export PYTHONPATH={}/radiotools:$PYTHONPATH \n'.format(software)

			header += 'export GSLDIR=/usr/lib'
			header += 'export ROOTSYS=/afs/ifh.de/group/radio/software/root5.34.36/compile\n'
			header += 'export PATH=$ROOTSYS/bin:$PATH\n'
			header += 'export PYTHONPATH=$ROOTSYS/lib:$PYTHONPATH\n'
			header += 'export LIBRARY_PATH=/afs/ifh.de/group/radio/software/anaconda_p3/lib:$LIBRARY_PATH\n'
			header += 'export LD_LIBRARY_PATH=/afs/ifh.de/group/radio/software/anaconda_p3/lib:$LD_LIBRARY_PATH\n'
			header += 'export LD_LIBRARY_PATH=$ROOTSYS/lib:$LD_LIBRARY_PATH\n'
			header += 'export SNS=/afs/ifh.de/group/radio/software/snowshovel/trunk\n'
			header += 'export LD_LIBRARY_PATH=$SNS/lib:$LD_LIBRARY_PATH\n'
			header += 'export PYTHONPATH=${SNS}/lib:$PYTHONPATH\n'
			header += 'export PYTHONPATH=$SNS:$PYTHONPATH\n'
			header += 'export PYTHONPATH=/afs/ifh.de/group/radio/software/radiotools:$PYTHONPATH\n'

			header += 'cd $TMPDIR \n'

			print(os.path.join(working_dir, 'run', 'run_'+os.path.basename(filename) + ".sh"))
			with open(os.path.join(working_dir, 'run', 'run_'+os.path.basename(filename) + ".sh"), 'w+') as fout:
				fout.write(header)
				fout.write(copy)
				fout.write(cmd)
				fout.write(copy_back)
