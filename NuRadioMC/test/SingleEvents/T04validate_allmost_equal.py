#!/usr/bin/env python3
import sys
from NuRadioMC.utilities._test_helpers import check, validate_hdf5_files, DEFAULT_RTOL, TIME_ATOL

checks = [
    check(['trigger_names', 'Tnoise', 'dt', 'thetamin', 'zmax', 'zmin', 'thetamax', 'fiducial_zmax',
           'fiducial_zmin', 'flavors', 'rmin', 'total_number_of_events', 'deposited', 'phimax', 'phimin',
           'Emin', 'rmax', 'fiducial_rmax', 'Emax', 'fiducial_rmin', 'n_events'], location='attrs'),
    check(['bandwidth', 'Vrms'], location='attrs', rtol=DEFAULT_RTOL),
    check(['azimuths', 'energies', 'event_group_ids', 'flavors', 'inelasticity', 'interaction_type',
           'multiple_triggers', 'n_interaction', 'triggered', 'xx', 'yy', 'zeniths', 'zz']),
    check(['weights'], rtol=DEFAULT_RTOL),
    check(['trigger_times'], atol=TIME_ATOL),
    check(['ray_tracing_solution_type'], location='station'),
    check(['ray_tracing_C0', 'ray_tracing_C1', 'launch_vectors', 'receive_vectors', 'travel_distances',
           'polarization'], location='station', rtol=DEFAULT_RTOL),
    check(['travel_times', 'trigger_times_per_event', 'trigger_times'], location='station', atol=TIME_ATOL),
    # signal amplitudes inherit platform-dependent differences of the ray-tracing solution, e.g. via
    # the numerical focusing factor (observed up to ~5e-4 for weak rays)
    check(['maximum_amplitudes_envelope', 'max_amp_shower_and_ray'], location='station', rtol=0.002),
    # may differ slightly because of differences in the interference between signals
    check(['maximum_amplitudes'], location='station', rtol=0.01),
]

validate_hdf5_files(sys.argv[1], sys.argv[2], checks, dump_on_failure=True)
