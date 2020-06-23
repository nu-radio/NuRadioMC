#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import os
from NuRadioReco.utilities import units
from NuRadioMC.EvtGen.generator import generate_surface_muons
import numpy as np

# define simulation volume
zmin = -3 * units.km  # the ice sheet at Summit Station is 3 km deep
zmax = 0 * units.km
rmin = 0 * units.km
rmax = 4 * units.km

thetamin = 58 * units.deg
thetamax = 62 * units.deg

path = os.path.dirname(os.path.abspath(__file__))

# generate one event list at 1e18 eV with 1000 atmospheric muons
generate_surface_muons(os.path.join(path,'1e18_full.hdf5'),
                       5e4, 1e18 * units.eV, 1e18 * units.eV,
                       rmin, rmax, zmin, zmax,
                       full_rmin=rmin, full_rmax=rmax,
                       full_zmin=zmin, full_zmax=zmax,
                       thetamin=thetamin, thetamax=thetamax,
                       config_file='NuRadioMC/test/atmospheric_Aeff/1e18eV/config_PROPOSAL_greenland.json')
