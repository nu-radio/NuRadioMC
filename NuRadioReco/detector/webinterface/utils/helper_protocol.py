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

det = Detector(database_connection=config.DATABASE_TARGET)


def check_inserted_config_information(cont, protocol):
    disabled = True

    # check if already in the database
    check_db = False
    protocols_db = load_measurement_protocols_from_db()
    if protocol not in protocols_db:
        check_db = True
    else:
        cont.info('The measurement protocol is already in the database.')

    # check that something is inserted
    inserted = False
    if protocol != '':
        inserted = True
    else:
        cont.error('Not all input filed are filled')

    if check_db and inserted:
        disabled = False

    return disabled


def load_measurement_protocols_from_db():
    return det.get_quantity_names('measurement_protocol', 'protocol')


def insert_measurement_protocol_into_db(protocol_name):
    det.add_measurement_protocol(protocol_name)
