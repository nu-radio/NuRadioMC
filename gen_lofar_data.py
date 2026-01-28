import logging
import lofar_processing
import multiprocessing

import pickle as pkl
import pandas as pd
import numpy as np

from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TITLE = "lofar-snr-7"
OUTPUT_DIR = Path(f"analysis/{TITLE}/data")
NUM_THREADS = 12
SNR = 7

def processing_wrapper(df_chunk):
    """Wrapper for the lofar_data_processing function to
    make the pipeline usable for multiprocessing.

    Args:
        df_chunk (pd.DataFrame): chunk of dataframe with events to analyse
    """
    sim_files = lofar_processing.get_sim_files(df_chunk)
    lofar_processing.lofar_data_processing(df_chunk, OUTPUT_DIR, sim_files, SNR)


with open("xmax_statistics_PAPER_2021.dat", "rb") as f:
    data = pkl.load(f, encoding="latin1")

df = pd.DataFrame.from_dict(data["events_passed_all"])
df.columns = [
    "event_id",
    "zenith",
    "azimuth",
    "core_x",
    "core_y",
    "d_ratio",
    "p_ratio",
    "energy",
    "xreco",
    "xreco_radio_only",
    "sigma_x",
    "sigma_e",
    "sigma_logE_radio",
    "std_core",
    "sigma_x_combined",
    "sigme_e_combined",
    "std_core_combined",
    17,
    18,
    19,
    "p_passed",
    "r_passed",
    "q_passed",
    "combchi2",
    "radiochi2",
    "nof_stations",
    "median_SNR",
    27,
    28,
]

filtered_df = df.loc[df["std_core"] < 10]
df_splits = np.split(filtered_df, NUM_THREADS)

pool = multiprocessing.Pool(NUM_THREADS)
pool.map(processing_wrapper, df_splits)

