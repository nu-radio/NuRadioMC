from NuRadioMC.EvtGen.generator import generate_eventlist_cylinder
from NuRadioReco.utilities import units
import numpy as np
import argparse

 
parser = argparse.ArgumentParser(
    description='Generation of neutrino events in a cylindrical volume around the station. The generated events will be stored in an HDF5 file, which can be used as input for the simulation step.'
)
parser.add_argument(
    'Number_of_events',
    type=int,
    help='Number of events to generate.'
)
parser.add_argument(
    'output_file',
    type=str,
    help='Name of the .nur file the simulated events will be written into.'
)

args = parser.parse_args()

#substitute the interested station position

# # Station absolute position
# station_x = -1572.81617365 * units.m  
# station_y =  729.37223865 * units.m  
# station_z =  -3.78323115 * units.m  

station_x = 0 * units.m  
station_y =  0 * units.m  
station_z =  0 * units.m  



# Generate event list for 1000 events at 10^19 eV
n_events = args.Number_of_events

# Define cylindrical volume centered on the station
zmin = station_z - 2.7 * units.km  
zmax = station_z          
rmin = 0 * units.km
rmax = 3.9 * units.km

# Energy range
Emin = 5e16 * units.eV
Emax = 1e19 * units.eV
# Flavor
flavor = [12]  # electron neutrino

# Generate the event list
generate_eventlist_cylinder(
    filename=args.output_file,
    n_events=n_events,
    Emin=Emin,
    Emax=Emax,
    volume={
        'fiducial_rmin': rmin,
        'fiducial_rmax': rmax,
        'fiducial_zmin': zmin,
        'fiducial_zmax': zmax,
        'x0': station_x,   # <-- shifts cylinder center horizontally
        'y0': station_y    # <-- to station position
    },
    flavor=flavor,
    n_events_per_file=None,
    start_event_id=0,
    interaction_type='nc',
    deposited=True
)