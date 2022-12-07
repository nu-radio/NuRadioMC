import streamlit as st
from plotly import subplots
import plotly.graph_objs as go
import pandas as pd
from NuRadioReco.detector.webinterface.utils.units_helper import str_to_unit
from NuRadioReco.utilities import units
import numpy as np
from NuRadioReco.detector.detector_mongo import Detector
from NuRadioReco.detector.webinterface import config
from datetime import datetime
from datetime import time

det = Detector(config.DATABASE_TARGET)
#det = Detector(database_connection='test')


def get_station_ids_from_db():
    return det.get_quantity_names('runtable', 'station')


def get_firmware_from_db():
    return det.get_quantity_names('runtable', 'firmware_version')


def load_runs(station_list, start_time, end_time, flag_list, trigger_list, min_duration, firmware_list):
    results = det.get_runs(station_list, start_time, end_time, flag_list, trigger_list, min_duration, firmware_list)
    return results