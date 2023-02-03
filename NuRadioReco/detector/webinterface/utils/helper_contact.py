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

# det = Detector(config.DATABASE_TARGET)
det = Detector(database_connection=config.DATABASE_TARGET)


def check_inserted_config_information(cont, name, email):
    disabled = True

    # validate email address
    validate_email = False
    allowed_email_endings = ['.com', '.de', '.edu', '.be', '.se', '.nl']
    if '@' in email and email[email.rfind('.'):] in allowed_email_endings:
        validate_email = True
    else:
        cont.error('The email address is not valid.')

    check_data_base = False
    db_info = load_contact_information_from_db(name, email)
    if db_info == []:
        check_data_base = True
    else:
        for dic in db_info:
            check_data_base = True
            if dic['name'] == name and dic['email'] == email:
                check_data_base = False
                cont.error('The contact information already exists in the database.')

    # check that something is inserted
    inserted = False
    if name != '' and email != '':
        inserted = True
    else:
        cont.error('Not all input filed are filled')

    if validate_email and check_data_base and inserted:
        disabled = False

    return disabled


def load_contact_information_from_db(name=None, email=None):
    return det.get_contact_information(contact_name=name, contact_email=email)


def insert_contact_information_into_db(name, email):
    det.add_contact_information(name, email)
