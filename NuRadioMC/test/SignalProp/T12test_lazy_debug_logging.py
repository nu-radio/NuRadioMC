#!/usr/bin/env python3
"""
Check that ``ray_tracing.set_start_and_end_point`` does not format numpy arrays for debug log
messages unless debug logging is enabled.

``set_start_and_end_point`` is called for every shower and channel. Formatting arrays into strings
eagerly (e.g. ``"{}".format(array)``) cost about as much as the ray tracing itself. The test counts
how often numpy formats a float via a print option formatter. Exits with a non-zero code on failure.
"""
import logging
import sys
import numpy as np

from NuRadioReco.utilities import units
from NuRadioMC.SignalProp import analyticraytracing as ray
from NuRadioMC.utilities import medium

n_formatted = [0]


def counting_formatter(value):
    n_formatted[0] += 1
    return str(value)


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


np.set_printoptions(formatter={'float_kind': counting_formatter})
ray_tracer = ray.ray_tracing(medium.southpole_simple(), log_level=logging.WARNING)
ray_logger = logging.getLogger('NuRadioMC.ray_tracing')
handler = ListHandler()
ray_logger.addHandler(handler)

errors = []

# debug logging disabled: no array may be formatted
ray_tracer.set_start_and_end_point(np.array([100., 20., -500.]) * units.m, np.array([0., 0., -100.]) * units.m)
if n_formatted[0] != 0:
    errors.append(f"{n_formatted[0]} floats of numpy arrays were formatted although debug logging is disabled")

# debug logging enabled: the messages still contain the positions
ray_logger.setLevel(logging.DEBUG)
ray_tracer.set_start_and_end_point(np.array([200., 20., -600.]) * units.m, np.array([0., 0., -100.]) * units.m)
messages = "\n".join(handler.messages)
for expected in ["X1 = [", "X2 = [", "dphi = ", "X1r = [", "2D points ["]:
    if expected not in messages:
        errors.append(f"debug output does not contain '{expected}':\n{messages}")

ray_logger.removeHandler(handler)
np.set_printoptions(formatter=None)

if errors:
    print("\n".join(errors))
    sys.exit(1)
print("ray tracing debug messages are only formatted when debug logging is enabled")
