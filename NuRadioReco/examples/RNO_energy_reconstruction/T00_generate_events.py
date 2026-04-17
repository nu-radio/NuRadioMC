from NuRadioMC.EvtGen.generator import generate_eventlist_cylinder
from NuRadioReco.utilities import units
import numpy as np

# Station absolute position
station_x = -1572.81617365 * units.m  
station_y =  729.37223865 * units.m  
station_z =  -3.78323115 * units.m  

# Generate event list for 1000 events at 10^19 eV
n_events = 1000

# Define cylindrical volume centered on the station
zmin = station_z - 0.5 * units.km   # ~-504 m depth
zmax = station_z                      # ~-3.78 m
rmin = 0 * units.km
rmax = 0.5 * units.km

# Energy range
Emin = 1e19 * units.eV
Emax = 1e19 * units.eV

# Flavor
flavor = [12]  # electron neutrino

# Generate the event list
generate_eventlist_cylinder(
    filename="1e19_final.hdf5",
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