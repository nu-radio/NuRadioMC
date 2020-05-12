"""
This file creates jobs for generating input files with secondary interactions.
It is meant to be used with Proposal > 6.1.1 and on local. The reason of having
this file is that the pip-installed version of Proposal seems to randomly stall
and create a fatal memory leak on the DESY Zeuthen cluster.

That doesn't mean we cannot encounter memory leaks on local! If we use the standard
input generating scripts, like the ones in examples/06_webinar/W01*py, sometimes
we will find out that the process also stalls for no reason. Even if we execute
the script and it succeeds, executing it again can cause the program to stall,
probably because of a Proposal-related problem with deleting allocated memory.

As a workaround, we will use the multiprocessing module to create a process and
explicitly kill it after a given timeout or after it ends. We repeat this procedure
to generate files for each energy bin.

To use it, type:

python T01_generate_proposal_local.py right_folder --init init

Where right_folder is an integer that controls the energy bin and init (optional)
defines the number of the first input file we will generate. So, to get files from
1 PeV to 100 EeV, we can do a bash loop:

for right_folder in {0..28}; do python T01_generate_local.py $right_folder; done

Be warned! This will take about a day. So maybe it's better to see if you can run
jobs on your favourite cluster with Proposal 6.1.1.
"""
from NuRadioReco.utilities import units
from NuRadioMC.EvtGen.generator import generate_eventlist_cylinder
import numpy as np
import os
import argparse
import multiprocessing as mp
import time

parser = argparse.ArgumentParser(description='Generation with Proposal')
parser.add_argument('right_folder', type=int,
                    help='right folder', default=0)
parser.add_argument('--init', type=int,
                    help='initial number', default=0)
args = parser.parse_args()

right_folder = args.right_folder
init = args.init


def generate(nevt, ipart, right_folder, queue, wait=0):

    """
    We define a fiducial volume for a 10x10 array with a spacing of 1.25 km, and
    a 3 km deep ice layer (Summit Station)
    """
    zmin = -3 * units.km
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
    full_zmin = -5 * units.km

    """
    We are going to generate the three neutrino and antineutrino flavours in an
    equal mixture.
    """
    flavours = [12, -12, 14, -14, 16, -16]

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
    imin = 0
    Emins = Es[0:-1]
    Emaxs = Es[1:]

    nevt_perfile = None

    """
    We generate nevt events per file and repeat the generation N times for each
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
    max_95_rmax = 95 * units.km

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

    	ifolder = -1
    	for folder, Emin, Emax in zip(folderlist, Emins, Emaxs):

    		ifolder += 1

    		if ifolder != right_folder:
    			continue
    		if (ifolder < imin):
    			continue
    		input_dir = os.path.join(folder_angle, folder, 'input')
    		try:
    			os.mkdir(input_dir)
    		except:
    			pass

    		cover_factor = 1.25
    		start_event_id = int(ipart * nevt * cover_factor * (max_95_rmax/rmax)**2 + 1)

    		outname = folder_angle+'/'+folder+'/input/'+folder+'.hdf5.part{:04d}'.format(ipart)
    		print(outname)
    		generate_eventlist_cylinder(outname, nevt, Emin, Emax, rmin, rmax, zmin, zmax,
    		full_zmin=full_zmin, thetamin=thetamin, thetamax=thetamax, phimin=phimin,
    		phimax=phimax, flavor=flavours, proposal_config=proposal_config,
    		n_events_per_file=nevt_perfile, deposited=False,
    		proposal=True, add_tau_second_bang=True,
    		start_event_id=start_event_id)

"""
This function is used to know how many fiducial events we want per file and how
many files we want for each energy, as a function of the folder number. As energy
increases, the number of secondary interactions increases as well and this takes
a massive toll on memory, so the numbers have to be lowered. Fortunately, at
high energies many more events trigger so we can have much better statistics
with fewer fiducial events.

The current get_nevt function works on my laptop, but maybe you'll have to change
it to get it to work on your computer.
"""
def get_nevt(i_folder):

    return (5000, 200)

    nevt_dict = { 20 : 1000, 21 : 850, 22 : 700, 23 : 600, 24 : 500, 25 : 400,
                  26 : 300, 27: 200, 28: 100}
    if i_folder < 11:
        return (5000, 200)
    elif i_folder < 18:
        return (2500, 400)
    elif i_folder < 20:
        return (1500, 600)
    else:
        nevt = nevt_dict[i_folder]
        nparts = int( 1000 * 250 / nevt )
        print("Number and parts:", nevt, nparts)
        return (nevt, nparts)

nevt, max_nparts = get_nevt(right_folder)

"""
Main part of the code. We loop on the different files.
"""
for ipart in range(init, max_nparts):

    print(f'{right_folder:d}, part {ipart:d}')

    keep_trying = True
    n_tries = 0

    """
    We create a timeout, which will be increased if the process crashes, until a
    max_timeout is reached.
    """
    timeout = 300
    max_timeout = 600

    """
    The multiprocessing module uses the same random seed every time a process
    is created. So, we need to manually change the numpy seed to get different
    results when executing Proposal.
    """
    seed = ipart + right_folder * 100000

    while keep_trying:

        if n_tries > 2:
            break

        queue = mp.Queue()

        """
        We create a process and pin it to the generate function.
        """
        proc = mp.Process(target = generate, args = (nevt, ipart, right_folder, queue))
        np.random.seed(seed) # We choose a random seed to get different events for each file
        proc.start()
        proc.join(timeout = timeout)
        if proc.is_alive():
            proc.terminate()
            print('Process failed.')
            n_tries += 1
            timeout += 300

            seed += 1

            if timeout > max_timeout:
                timeout = max_timeout
        else:
            print('Process successful!')
            proc.terminate()
            keep_trying = False
