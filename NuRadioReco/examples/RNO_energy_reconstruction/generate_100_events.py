from NuRadioMC.EvtGen.generator import generate_eventlist_cylinder
from NuRadioReco.utilities import units
import numpy as np

# Generate event list for 100 events at 10^19 eV
n_events = 1000

# Define cylindrical volume
# Radius and height of cylinder around detector
zmin = -4 * units.km
zmax = 0 * units.km
rmin = 0 * units.km
rmax = 3 * units.km

# Energy range
Emin = 1e19 * units.eV
Emax = 1e19 * units.eV  # Fixed energy

# Flavor
flavor = [12]  # electron neutrino

# Generate the event list
print(f"Generating {n_events} neutrino events at E = 10^19 eV...")
generate_eventlist_cylinder(
    filename="1e19_n1000.hdf5",
    n_events=n_events,
    Emin=Emin,
    Emax=Emax,
    volume={
        'fiducial_rmin': rmin,
        'fiducial_rmax': rmax,
        'fiducial_zmin': zmin,
        'fiducial_zmax': zmax
    },
    flavor=flavor,
    n_events_per_file=None,
    start_event_id=0,
    deposited=True
)

print(f"Event list saved to: 1e19_n100.hdf5")
print(f"Number of events: {n_events}")