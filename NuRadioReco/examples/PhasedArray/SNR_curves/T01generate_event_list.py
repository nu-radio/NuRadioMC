"""
This file generates the input files for the electron neutrino SNR
curves simulation using a phased array. For creating a test file, run:

python T01generate_event_list.py

The file will create several folders containing neutrinos that interact at a
given angle with respect to the horizon. The

WARNING: This file needs NuRadioMC installed. https://github.com/nu-radio/NuRadioMC
"""

from __future__ import absolute_import, division, print_function
from NuRadioReco.utilities import units
from NuRadioMC.EvtGen.generator import generate_eventlist_cylinder
import numpy as np
import os

z12 = -200 * units.m
rho = 500 * units.m

obs_angles = np.linspace(-10.,10.,41) * units.deg
# Elevation of the vertex point seen from the middle point of the array

for obs_angle in obs_angles:
	# define simulation volume
	zmin = z12 + rho*np.tan(obs_angle) - 0.1 * units.m
	zmax = z12 + rho*np.tan(obs_angle)
	rmin = rho
	rmax = rho + 0.1 * units.m
	print('Neutrino height', zmax)

	theta_c = 55 * units.deg
	theta_nu = 90*units.deg - theta_c - obs_angle

	phimin = 0.*units.deg
	phimax = 0.1*units.deg

	if (theta_nu < 0):
		theta_nu = -theta_nu
		phimin = 180*units.deg
		phimax = 180.1*units.deg

	thetas = np.array([theta_nu, theta_nu + 0.01*units.deg])
	thetamins = thetas[0:-1]
	thetamaxs = thetas[1:]
	print(thetamins)
	print(thetamaxs)

	Es = np.array([18,18.1])
	Es = 10**Es
	Emins = Es[0:-1]
	Emaxs = Es[1:]

	flavours = [12] # no difference between neutrinos and antineutrinos for us

	nevt = 2e3
	nevt_perfile = 1e3

	for thetamin, thetamax in zip(thetamins, thetamaxs):

		for flavour in flavours:

			folder_angle = "{:.1f}deg_{}".format(thetamin/units.deg,flavour)
			try:
				os.mkdir(folder_angle)
			except:
				pass

			folderlist = []
			for ifolder, Emin,Emax in zip(range(len(Emins)),Emins,Emaxs):
				folderlist.append("{:.2f}_{}_{}_{:.2e}_{:.2e}".format(thetamin/units.deg,flavour,str(ifolder).zfill(2),Emin,Emax))
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
				print(Emin/units.PeV, Emax/units.PeV)
				print(outname)
				generate_eventlist_cylinder(outname, nevt, Emin, Emax, rmin, rmax, zmin, zmax, full_rmin=rmin, full_rmax=rmax, full_zmin=zmin, full_zmax=zmax, thetamin=thetamin, thetamax=thetamax, phimin=phimin, phimax=phimax, start_event_id=1, flavor=[flavour], n_events_per_file=nevt_perfile, deposited=True)
