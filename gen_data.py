"""Script that analyses events using multithreading to speed up analysis. Constants
at the top of the script should be changed to relevant values for analysis run.
LIST_OR_NUM_EVENTS should either be "list" or "num", this will determine if script
uses a random number of events in the HDF5_directory
"""

import logging
import multiprocessing
import pipeline

import numpy as np

from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
mlogger = multiprocessing.log_to_stderr(level=logging.INFO)
# CONSTANTS, DON'T FORGET TO CHANGE THEM TO RELEVANT RUN
TITLE = "prelimenary_results_7"
STATIONS = [1, 2, 3, 4, 5, 6, 7]
NUM_THREADS = 14

LIST_OR_NUM_EVENTS = "list"  # list for fixed event list or num of events needed
# EVENT_IDS = [86132542, 132649890, 268490510, 219294520]
EVENT_IDS = [99394319, 95591951, 86132542, 83827322, 74486916,
        70848963, 69134130, 66187840, 63934386, 336673661,
        285360413, 273524336, 268490510, 260691538, 242151059,
        237959892, 237608273, 234548441, 230943673, 225865341,
        219294520, 217006485, 212089170, 205642073, 198490827,
        190669040, 183985109, 181699538, 178027768, 175038176,
        172055925, 161326774, 157571174, 149538803, 142043645,
        131373704, 132649890, 127437284, 126346457, 123314680,
        111394500, 105147553]
NUM_EVENTS = 288
OUTPUT_DIR = Path(f"analysis/{TITLE}")


def get_data_from_ids(event_ids: list):
    """Function is a helper that takes a list of event_ids
    to analyse. Useful for multiprocessing purposes.

    Args:
        event_ids (list): list of event_ids
    """
    fps = pipeline.get_filepaths(event_ids, 15)
    pipeline.generate_data(fps[0], fps[1], OUTPUT_DIR, STATIONS)


# -------------ANALYSIS---------------
if LIST_OR_NUM_EVENTS == "list":
    event_ids = np.split(np.array(EVENT_IDS), NUM_THREADS)
elif LIST_OR_NUM_EVENTS == "num":
    selection_events = pipeline.select_events_to_analyse(NUM_EVENTS)
    event_ids = np.split(selection_events, NUM_THREADS)
else:
    raise ValueError(
        """LIST_NUM_OR_EVENTS should either be "list" or "num" but was neither"""
    )

logger.info(f"Processing the following events: {event_ids}")

pool = multiprocessing.Pool(NUM_THREADS)
pool.map(get_data_from_ids, event_ids)


# --------------PLOTTING--------------
LOFAR_DATA_DIR = OUTPUT_DIR / "data" / "LOFAR"
STAR_DATA_DIR = OUTPUT_DIR / "data" / "star"
PLOT_DIR = OUTPUT_DIR / "plots"

logger.info("Making general plots")

lofar_df = pipeline.read_events_from_pkl(LOFAR_DATA_DIR)
star_df = pipeline.read_events_from_pkl(STAR_DATA_DIR)

lofar_full_max_rit = pipeline.full_plot(lofar_df, TITLE)
star_full_max_rit = pipeline.full_plot(star_df, TITLE)
lofar_full_max_rit.savefig(PLOT_DIR / "lofar_full_rit.png")
star_full_max_rit.savefig(PLOT_DIR / "star_full_rit.png")

lofar_3d_fig = pipeline.three_dim_plot(lofar_df)
star_3d_fig = pipeline.three_dim_plot(star_df)
lofar_3d_fig.write_html(PLOT_DIR / "lofar_3d.html")
star_3d_fig.write_html(PLOT_DIR / "star_3d.html")

lofar_hist = pipeline.diagnostic_hist(lofar_df)
star_hist = pipeline.diagnostic_hist(star_df)
lofar_hist.savefig(PLOT_DIR / "lofar_hist.png")
star_hist.savefig(PLOT_DIR / "star_hist.png")

logger.info("Succesfully finished generating data")
