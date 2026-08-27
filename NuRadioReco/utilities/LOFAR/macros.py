'''Macros used for LOFAR data analysis.'''

import datetime
import numpy as np

# default stations used for LOFAR data analysis
DEFAULT_STATIONS = [
    "CS001", "CS002", "CS003", "CS004", "CS005", "CS006", "CS007",
    "CS011", "CS013", "CS017",
]

# paths for data events
TBB_DIRECTORY = "/vol/astro5/lofar/astro3/vhecr/lora_triggered/data/"
JSON_DIRECTORY = "/vol/astro7/lofar/kratos_files/json"
META_DATA_DIRECTORY = "/vol/astro7/lofar/vhecr/kratos/data/"
BLOCK_NUMBER_FILE = "/vol/astro5/lofar/astro3/vhecr/lora_triggered/LORA/LORAtime4"
GDAS_ATMOSPHERE_DIRECTORY = "/vol/astro7/lofar/sim/pipeline/atmosphere_files"

# paths for simulated events
COREAS_PARENT_HDF5_DIRECTORY = "/vol/astro7/lofar/sim/hdf5_files"  
ANTENNA_RESPONSE_DIRECTORY = "something"  # TODO: set this to the correct path for antenna response data in Radboud

# some hard coded values in the base code, can be directly modified here
DATA_TRACE_LENGTH = 65536
RFI_CLEANING_TRACE_LENGTH = 8192
CR_SNR = 6.5
PASS_BAND = (30, 80)
START_TIME = datetime.datetime(2012, 10, 1, 0, 0)

# LORA parameters
LORA_CORE_PRECISION = 30.0
LORA_ANGLE_PRECISION = np.radians(0.7)