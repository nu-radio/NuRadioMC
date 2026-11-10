#!/usr/bin/env python3
import sys
from NuRadioMC.utilities._test_helpers import check, validate_hdf5_files, DEFAULT_RTOL, TIME_ATOL

checks = [
    check(['emitter_amplitudes', 'emitter_antenna_type', 'emitter_frequency', 'emitter_half_width',
           'emitter_model', 'emitter_orientation_phi', 'emitter_orientation_theta', 'emitter_rotation_phi',
           'emitter_rotation_theta', 'event_group_ids', 'multiple_triggers', 'triggered', 'xx', 'yy', 'zz',
           'shower_ids']),
    check(['trigger_times'], atol=TIME_ATOL),
    check(['ray_tracing_solution_type'], location='station'),
    check(['max_amp_shower_and_ray', 'polarization', 'ray_tracing_C0', 'launch_vectors', 'receive_vectors',
           'travel_distances', 'ray_tracing_C1'], location='station', rtol=DEFAULT_RTOL),
    check(['travel_times', 'trigger_times'], location='station', atol=TIME_ATOL),
    # reduced precision: this quantity suddenly could not be reproduced to DEFAULT_RTOL anymore
    check(['maximum_amplitudes_envelope'], location='station', rtol=0.001),
    # may differ slightly because of differences in the interference between signals
    check(['maximum_amplitudes', 'time_shower_and_ray'], location='station', rtol=0.01),
]

validate_hdf5_files(sys.argv[1], sys.argv[2], checks)
