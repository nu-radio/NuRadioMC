from NuRadioReco.modules.io.RNO_G import readRNOGDataMattak
from NuRadioReco.modules.io import eventWriter
from NuRadioReco.utilities import units

import sys
import logging

""" Usage: Reading mattak rootified data

`python read_rnog.py [INPUTFILES ...] [OUTPUTFILE]`
"""

if len(sys.argv) < 3:
    sys.exit("Error: You have to specify at least 2 arguments: One input path "
             "(to a mattak run directory or combined root file), and one "
             "output path (to the nur file which is created)")

list_of_root_files = sys.argv[1:-1]
output_filename = sys.argv[-1]

rnog_reader = readRNOGDataMattak.readRNOGData(log_level=logging.INFO)
writer = eventWriter.eventWriter()

"""
With a selector you can select or reject events based on information in the
Mattak class EventInfo. See https://github.com/RNO-G/mattak/blob/main/py/mattak/Dataset.py

class EventInfo:
    eventNumber: int
    station : int
    run: int
    readoutTime : float
    triggerTime : float
    triggerType: str
    sysclk: int
    sysclkLastPPS: Tuple[int, int]  # the last 2 PPS sysclks, most recent first
    pps: int
    radiantStartWindows: numpy.ndarray
    sampleRate: float  # Sample rate, in GSa/s
"""

# The following selector selects only events with a forced trigger.
selectors = [lambda einfo: einfo.triggerType == "FORCE"]

rnog_reader.begin(
    list_of_root_files,
    selectors=selectors,
    # Currently false because Mattak does not contain calibrated data yet
	read_calibrated_data=False,
 	# Only used when read_calibrated_data==False, performs a simple baseline subtraction each 128 bins
	apply_baseline_correction="approximate",
 	# Only used when read_calibrated_data==False, performs a linear voltage calibration with hardcoded values
	convert_to_voltage=True,
	# Can be used instead of defining a selector (only for triggers)
	select_triggers=None,
	# If true, and if the RunTable database is available select runs based on the following criteria
	select_runs=True,
	# Only use runs of a certain run type
	run_types=["physics"],
	# Only use runs with a maximum trigger rate of 1 Hz
	max_trigger_rate=1 * units.Hz,
    # This might be necessary to read older files which have an invalid value stored as sampling rate
    overwrite_sampling_rate=3.2
)

writer.begin(filename=output_filename)

for i_event, event in enumerate(rnog_reader.run()):
    writer.run(event)

rnog_reader.end()
writer.end()
