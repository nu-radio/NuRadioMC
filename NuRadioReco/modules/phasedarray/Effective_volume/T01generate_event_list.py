from __future__ import absolute_import, division, print_function
from NuRadioMC.utilities import units
from NuRadioMC.EvtGen.generator import generate_eventlist_cylinder
import numpy as np
import os

# define simulation volume
zmin = -2.7 * units.km  # the ice sheet at South Pole is 2.7km deep
zmax = 0 * units.km
rmin = 0 * units.km
rmax = 4 * units.km

costhetas = np.linspace(0.999, -1, 2) # Usually 20
thetas = np.arccos(costhetas)
#thetas = np.linspace(0.*units.deg, 180.*units.deg, 3)
thetamins = thetas[0:-1]
thetamaxs = thetas[1:]
print(thetamins)
print(thetamaxs)

phimin = 0.*units.deg
phimax = 360.*units.deg

logEs = np.linspace(15., 20., 50) # Usually 50
Es = 10**logEs * units.eV
Emins = Es[0:-1]
Emaxs = Es[1:]

flavours = [12] # no difference between neutrinos and antineutrinos for us
                    # also taus and mus are identical in the present version of the code
nevt = 1e6
nevt_perfile = 1e5

for thetamin, thetamax in zip(thetamins, thetamaxs):

	for flavour in flavours:

		folder_angle = "{:.2f}rad_{}".format(thetamin,flavour)
		try:
			os.mkdir(folder_angle)
		except:
			pass

		folderlist = []
		for ifolder, Emin,Emax in zip(range(len(Emins)),Emins,Emaxs):
			folderlist.append("{:.2f}_{}_{}_{:.2e}_{:.2e}".format(thetamin,flavour,str(ifolder).zfill(2),Emin,Emax))
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

# generate one event list at 1e19 eV with 100 neutrinos
#generate_eventlist_cylinder('1e19_n1e2.hdf5', 1e2, 1e19 * units.eV, 1e19 * units.eV, rmin, rmax, zmin, zmax)

# generate one event list at 1e19 eV with 1000 neutrinos
#generate_eventlist_cylinder('1e19_n1e3.hdf5', 1e3, 1e19 * units.eV, 1e19 * units.eV, rmin, rmax, zmin, zmax)

# generate one event list at 1e18 eV with 10000 neutrinos
#generate_eventlist_cylinder('1e18_n1e4.hdf5', 1e4, 1e18 * units.eV, 1e18 * units.eV, rmin, rmax, zmin, zmax)
