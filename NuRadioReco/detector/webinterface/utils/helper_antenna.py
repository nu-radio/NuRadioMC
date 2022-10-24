import streamlit as st
from plotly import subplots
import plotly.graph_objs as go
import pandas as pd
from NuRadioReco.detector.webinterface.utils.units_helper import str_to_unit
from NuRadioReco.utilities import units
import numpy as np
# from NuRadioReco.detector.detector_mongo import det
from NuRadioReco.detector.detector_mongo import Detector
from NuRadioReco.detector.webinterface import config
# from NuRadioReco.detector.detector_mongo import Detector
from datetime import datetime

# det = Detector(database_connection='env_pw_user')
# det = Detector(database_connection='test')
det = Detector(config.DATABASE_TARGET)

def select_antenna_name(antenna_type, container, warning_container):
    selected_antenna_name = ''
    # update the dropdown menu
    antenna_names = det.get_object_names(antenna_type)
    antenna_names.insert(0, f'new {antenna_type}')

    col1, col2 = container.columns([1, 1])
    antenna_dropdown = col1.selectbox('Select existing antenna or enter unique name of new antenna:', antenna_names)
    # checking which antenna is selected
    if antenna_dropdown == f'new {antenna_type}':
        # if new antenna is chosen: The text input will be enabled
        disable_new_antenna_name = False
    else:
        # Otherwise the chosen antenna will be used and warning will be given
        disable_new_antenna_name = True
        selected_antenna_name = antenna_dropdown
        warning_container.warning(f'You are about to override the {antenna_type} unit {antenna_dropdown}!')
    antenna_text_input = col2.text_input('', placeholder=f'new unique {antenna_type} name', disabled=disable_new_antenna_name)
    if antenna_dropdown == f'new {antenna_type}':
        selected_antenna_name = antenna_text_input

    return selected_antenna_name, antenna_dropdown, antenna_text_input


def validate_global(page_name, container_bottom, antenna_name, new_antenna_name, channel_working, Sdata_validated, uploaded_data):
    disable_insert_button = True
    name_validation = False
    # if nothing is chosen, a warning is given and the INSERT button stays disabled
    if antenna_name == '':
        container_bottom.error('Antenna name is not set')
    elif antenna_name == f'new {page_name}' and (new_antenna_name is None or new_antenna_name == ''):
        container_bottom.error(f'Antenna name dropdown is set to \'new {page_name}\', but no new antenna name was entered.')
    else:
        name_validation = True

    if name_validation:
        if not Sdata_validated and uploaded_data is not None:
            container_bottom.error('There is a problem with the input data')
            disable_insert_button = True
        elif Sdata_validated:
            disable_insert_button = False
            container_bottom.success('Input fields are validated')

        if not channel_working:
            container_bottom.warning('The channel is set to not working')
            disable_insert_button = False
            container_bottom.success('Input fields are validated')
    else:
        disable_insert_button = True

    return disable_insert_button


def insert_to_db(page_name, s_name, antenna_name, data, working, primary, protocol, input_units):
    if not working:
        det.set_not_working(page_name, antenna_name, primary)
    else:
        det.antenna_add_Sparameter(page_name, antenna_name, [s_name], data, primary, protocol, input_units)
