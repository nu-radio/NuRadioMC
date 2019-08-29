from __future__ import absolute_import, division, print_function
from NuRadioMC.utilities import units
from NuRadioMC.EvtGen.generator import generate_eventlist_cylinder
import os
import sys

"""
This file creates four event files to be used for the coincidence analysis.
The energies and the number of events can be changed below.

The working directory can be passed as an argument. By default, the current
directory is used.
"""

# define simulation volume
zmin = -2.7 * units.km  # the ice sheet at South Pole is 2.7km deep
zmax = 0 * units.km
rmin = 0 * units.km
rmax = 4 * units.km

try:
    working_dir = sys.argv[1]
except:
    print("Usage python A01generate_event_list.py working_dir")
    print("Using current directory as working directory")
    working_dir = "."

if not os.path.exists(os.path.join(working_dir, "event_files")):
    os.makedirs(os.path.join(working_dir, "event_files"))

# generate one event list at 1e18 eV with 10000 neutrinos
generate_eventlist_cylinder(os.path.join(working_dir, '1e17_n1e5.hdf5'), 1e4, 1e17 * units.eV, 1e17 * units.eV, rmin, rmax, zmin, zmax,
                            full_rmin=rmin, full_rmax=rmax, full_zmin=zmin, full_zmax=zmax)

generate_eventlist_cylinder(os.path.join(working_dir, '1e18_n1e5.hdf5'), 1e4, 1e18 * units.eV, 1e18 * units.eV, rmin, rmax, zmin, zmax,
                            full_rmin=rmin, full_rmax=rmax, full_zmin=zmin, full_zmax=zmax)

generate_eventlist_cylinder(os.path.join(working_dir, '1e19_n1e5.hdf5'), 1e4, 1e19 * units.eV, 1e19 * units.eV, rmin, rmax, zmin, zmax,
                            full_rmin=rmin, full_rmax=rmax, full_zmin=zmin, full_zmax=zmax)

generate_eventlist_cylinder(os.path.join(working_dir, '1e20_n1e5.hdf5'), 1e4, 1e20 * units.eV, 1e20 * units.eV, rmin, rmax, zmin, zmax,
                            full_rmin=rmin, full_rmax=rmax, full_zmin=zmin, full_zmax=zmax)
