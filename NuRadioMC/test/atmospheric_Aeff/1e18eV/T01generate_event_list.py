#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import os
from NuRadioReco.utilities import units
from NuRadioMC.EvtGen.generator import generate_surface_muons

volume = {
'fiducial_rmin':0 * units.km,
'fiducial_rmax': 4 * units.km,
'fiducial_zmin':-3 * units.km,  # the ice sheet at Summit Station is 3 km deep
'fiducial_zmax': 0 * units.km}

# volume = {
# 'fiducial_xmin':-4 * units.km,
# 'fiducial_xmax': 4 * units.km,
# 'fiducial_ymin':-4 * units.km,
# 'fiducial_ymax': 4 * units.km,
# 'fiducial_zmin':-3 * units.km,  # the ice sheet at Summit Station is 3 km deep
# 'fiducial_zmax': 0 * units.km,
# 'full_xmax': 5 * units.km,
# 'full_xmin':-5 * units.km,
# 'full_ymax': 5 * units.km,
# 'full_ymin':-5 * units.km,
# 'full_zmin':-3 * units.km,
# 'full_zmax': 0 * units.km,
# }

thetamin = 58 * units.deg
thetamax = 62 * units.deg

path = os.path.dirname(os.path.abspath(__file__))

# generate one event list at 1e18 eV with 1000 atmospheric muons
generate_surface_muons(os.path.join(path, '1e18_full.hdf5'),
                       2.5e3, 1e18 * units.eV, 1e18 * units.eV,
                       volume,
                       thetamin=thetamin, thetamax=thetamax,
                       config_file=os.path.join(path, 'config_PROPOSAL_greenland.json'))
