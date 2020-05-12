"""
This file creates neutrino input files with primary interactions only.

To use it, type:

python T01_generate_primary.py
"""
from NuRadioReco.utilities import units
from NuRadioMC.EvtGen.generator import generate_eventlist_cylinder
import numpy as np
import os
import argparse

"""
We define a fiducial volume for a 10x10 array with a spacing of 1.25 km, and
a 3 km deep ice layer (Summit Station)
"""
zmin = -3 * units.km
zmax = 0 * units.km
rmin = 0 * units.km
rmax = 11 * units.km

"""
We are going to generate the three neutrino and antineutrino flavours in an
equal mixture.
"""
flavours = [12, -12, 14, -14, 16, -16]

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

"""
Let us generate one million events per energy bin and split each bin into
ten files with 50 000 events each. When using only one station, we could
request 100 000 events or more, but if you're using the RNO array or a
10x10 array, it's better to use smaller files.
"""
nevt = 1e6
nevt_perfile = 5e4

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
		print(outname)
		generate_eventlist_cylinder(outname, nevt, Emin, Emax, rmin, rmax, zmin, zmax,
            		thetamin=thetamin, thetamax=thetamax, phimin=phimin, phimax=phimax,
                    flavor=flavours, n_events_per_file=nevt_perfile, deposited=False,
            		proposal=False)
