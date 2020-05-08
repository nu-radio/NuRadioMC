"""
This file creates jobs for generating input files with secondary interactions.
These files should work on the DESY Zeuthen cluster. Make sure the paths
to NuRadioMC and NuRadioReco point to the right ones in your account.

This file uses an old version of Proposal with python2.7. The reason is that the
new Proposal version stalls randomly and presents a fatal memory leak when
executed of the DESY cluster. Should you prefer the new version, we recommend
you to use your local computer (armed also with patience) or another cluster.
The compiled library can be found in /afs/ifh.de/group/radio/software/proposal_lib

See T01_create_secondaries.py to use Proposal 6.1.1
"""
from NuRadioReco.utilities import units
from NuRadioMC.EvtGen.generator import generate_eventlist_cylinder
import numpy as np
import os

"""
We define a fiducial volume for a 10x10 array with a spacing of 1.25 km, and
a 3 km deep ice layer (Summit Station)
"""
zmin = -3. * units.km
zmax = 0 * units.km
rmin = 0 * units.km
rmax = 11 * units.km

"""
We increase our full_zmin to -3.5 km so that we include leptons generated in
rock that are then propagated to the ice and create showers in it. They rarely
trigger and don't have much weight because they're upgoing, though.

The option add_tau_second_bang will increase the full_rmax so that we can have
secondary interactions in our fiducial volume at high energies.
"""
full_zmin = -3.5 * units.km

"""
We are going to generate the three neutrino and antineutrino flavours in an
equal mixture.
"""
flavours = [12, -12, 14, -14, 16,-16]

"""
The Proposal configuration used is the one for Greenland. You can change it
to SouthPole or MooresBay. Make sure to change the table path in the config
file (in EvtGen) or write the path to your own Proposal config file.
"""
proposal_config = 'Greenland'

"""
We have chosen the whole sky as our zenith band. For point source studies, we
must use more zenith bands, preferably with the same solid angle. Changing the
2 to N in the np.linspace function below creates N-1 zenith bands in the sky
covering the same amount of solid angle. Our detector is thought to have azimuthal
symmetry, but a good exercise would be to check it if it actually. We could
study that generating phimin and phimax bands.
"""
costhetas = np.linspace(1, -1, 2)
thetas = np.arccos(costhetas)
thetamins = thetas[0:-1]
thetamaxs = thetas[1:]

phimin = 0.*units.deg
phimax = 360.*units.deg

"""
We generate 29 energy bins from 1 PeV to 100 EeV.
"""
logEs = np.linspace(15., 20., 30)
Es = 10**logEs * units.eV
Emins = Es[0:-1]
Emaxs = Es[1:]

"""
We generate 5000 events per file and repeat the generation 200 times for each
energy bin. The reason we create multiple files like this instead of creating
a single file and then splitting it into smaller files is because of the large
memory requirements it would have.

Since we are creating different files, we need to make sure that the event
indices don't overlap. That's why we assume a maximum volume increase to
95 kilometres. The add_tau_second_bang = True option will increase our volume
to include secondary interactions, and it will also increase the number of
input neutrinos by the same factor. So, we assume that the maximum volume is
95 km, and therefore the maximum number of events a file can have is
nevt * (max_95_rmax/rmax)**2, which we can use to create our start event index
for each file.
"""
nevt = 5e3
nevt_perfile = None
max_95_rmax = 95 * units.km
nparts = 200

"""
This loop will create a folder for each zenith band. Within those folders, each
energy bin will have their own folder. In each of the energy folders, we will
have files named job_input_XX.sh that we can submit to the queue. If they
succeed, the output file will be written in the 'input' folder on the energy
bin folder.
"""

"""
Change the software path to have your own software folder that contains
NuRadioMC, NuRadioReco and radiotools.
"""
software = '/afs/ifh.de/group/radio/scratch/garcia/software/'

for thetamin, thetamax in zip(thetamins, thetamaxs):


	folder_angle = "{:.2f}rad_all".format(thetamin)
	try:
		os.mkdir(folder_angle)
	except:
		pass

	folderlist = []
	for ifolder, Emin,Emax in zip(range(len(Emins)),Emins,Emaxs):
		folderlist.append("{:.2f}_{}_{}_{:.2e}_{:.2e}".format(thetamin,"all",str(ifolder).zfill(2),Emin,Emax))
	for folder in folderlist:
		try:
			os.mkdir(os.path.join(folder_angle, folder))
		except:
			pass

	for folder, Emin, Emax in zip(folderlist, Emins, Emaxs):

		input_dir = os.path.join(folder_angle, folder, 'input')
		try:
			os.mkdir(input_dir)
		except:
			pass
		outname = folder_angle+'/'+folder+'/input/'+folder+'.hdf5'
		outname_short = folder+'.hdf5'
		outname_full = os.path.join(os.path.dirname(os.path.abspath(__file__)), outname)

		for ipart in range(nparts):

			start_event_id = int(ipart * nevt * (max_95_rmax/rmax)**2 + 1)

			outname_full = os.path.join(os.path.dirname(os.path.abspath(__file__)), outname+'.part{:04d}'.format(ipart))
			outname_base = outname_short+'.part{:04d}'.format(ipart)
			python_filename = 'job_input_{:02d}.py'.format(ipart)
			sh_filename = 'job_input_{:02d}.sh'.format(ipart)
			with open(os.path.join(folder_angle,folder,python_filename), 'w') as f:
				f.write('from NuRadioMC.EvtGen.generator import generate_eventlist_cylinder\n')

				instruction  = "generate_eventlist_cylinder('{}', {}, {}, {}, {}, {}, {}, {},\n".format(outname_base, nevt, Emin, Emax, rmin, rmax, zmin, zmax)
				instruction += "full_zmin={}, thetamin={}, thetamax={}, phimin={},\n".format(full_zmin, thetamin, thetamax, phimin)
				instruction += "phimax={}, flavor={},\n".format(phimax, flavours)
				instruction += "n_events_per_file={}, deposited=False,\n".format(nevt_perfile)
				instruction += "start_event_id={}\n,".format(start_event_id)
				instruction += "proposal=True, add_tau_second_bang=True, proposal_config='{}')\n".format(proposal_config)
				f.write(instruction)

			full_folder = os.path.abspath(os.path.join(folder_angle, folder))
			with open( os.path.join(full_folder, sh_filename), 'w+' ) as f:

				header = '#!/bin/zsh\n'
				header += '#$ -S /bin/zsh\n'
				header += '#$ -l h_cpu=00:30:00\n'
				header += '#$ -l h_rss=1G\n'
				#header += '#$ -l tmpdir_size=10G\n'
				#(stderr and stdout are merged together to stdout)
				header += '#$ -j y\n'
				###(send mail on job's begin, end and abort bea) bea
				header += '#$ -m a\n'
				# execute job from current directory and not relative to your home directory
				header += '##$ -cwd\n'
				# send output files to the trash, COMMENT TO SEE THE OUTPUT AND DEBUG
				header += '#$ -o {}\n'.format(full_folder)
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
				header += 'export LIBRARY_PATH=/afs/ifh.de/group/radio/software/anaconda_p2/lib:$LIBRARY_PATH\n'
				header += 'export LD_LIBRARY_PATH=/afs/ifh.de/group/radio/software/anaconda_p2/lib:$LD_LIBRARY_PATH\n'
				header += 'export LD_LIBRARY_PATH=$ROOTSYS/lib:$LD_LIBRARY_PATH\n'
				header += 'export SNS=/afs/ifh.de/group/radio/software/snowshovel/trunk\n'
				header += 'export LD_LIBRARY_PATH=$SNS/lib:$LD_LIBRARY_PATH\n'
				header += 'export PYTHONPATH=${SNS}/lib:$PYTHONPATH\n'
				header += 'export PYTHONPATH=$SNS:$PYTHONPATH\n'
				header += 'export PYTHONPATH=/afs/ifh.de/group/radio/software/radiotools:$PYTHONPATH\n'
				header += 'export PYTHONPATH=/afs/ifh.de/group/radio/software/proposal_lib:$PYTHONPATH\n'

				header += 'cd $TMPDIR \n'

				f.write(header)

				instructions  = 'cp {} .\n'.format(os.path.join(full_folder,python_filename))
				instructions += '/afs/ifh.de/group/radio/software/anaconda_p2/bin/python2.7 {}\n'.format(python_filename)
				instructions += 'cp *hdf5* {}\n'.format(os.path.join(full_folder,'input'))

				f.write(instructions)
                                      
