#!/usr/bin/env python3
import argparse
from NuRadioMC.utilities._test_helpers import check, validate_hdf5_files, TIME_ATOL

parser = argparse.ArgumentParser(description='Compare two NuRadioMC hdf5 output files')
parser.add_argument('file1')
parser.add_argument('file2', help='reference file')
parser.add_argument('--rtol', type=float, default=1e-4,
                    help='relative tolerance of all keys which are not compared exactly or with an absolute tolerance')
args = parser.parse_args()

checks = [
    check(['trigger_names', 'Tnoise', 'dt', 'thetamin', 'zmax', 'zmin', 'thetamax', 'fiducial_zmax',
           'fiducial_zmin', 'flavors', 'rmin', 'total_number_of_events', 'deposited', 'phimax', 'phimin',
           'Emin', 'rmax', 'fiducial_rmax', 'Emax', 'fiducial_rmin', 'n_events'], location='attrs'),
    check(['bandwidth', 'Vrms'], location='attrs', rtol=args.rtol),
    check(['azimuths', 'energies', 'event_group_ids', 'flavors', 'inelasticity', 'interaction_type',
           'multiple_triggers', 'n_interaction', 'triggered', 'xx', 'yy', 'zeniths', 'zz']),
    check(['weights'], rtol=args.rtol),
    check(['trigger_times'], atol=TIME_ATOL),
    check(['ray_tracing_solution_type'], location='station'),
    check(['ray_tracing_C0', 'ray_tracing_C1', 'launch_vectors', 'receive_vectors', 'travel_distances',
           'polarization', 'max_amp_shower_and_ray', 'maximum_amplitudes', 'maximum_amplitudes_envelope'],
          location='station', rtol=args.rtol),
    check(['travel_times', 'trigger_times_per_event', 'trigger_times'], location='station', atol=TIME_ATOL),
]

validate_hdf5_files(args.file1, args.file2, checks, dump_on_failure=True)
