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

# det = Detector(config.DATABASE_TARGET)
det = Detector(database_connection='test')


def load_station_infos(station_id, coll_name):
    det.update(datetime.now(), coll_name)
    station_info = det.get_station_information(coll_name, station_id)

    return station_info


def build_collection_input(cont):
    col1, col2 = cont.columns([1, 1])

    # get all station collections
    collection_names = det.get_collection_names()
    sta_coll_names = []
    list_help = []
    for coll in collection_names:
        if 'station' in coll and 'trigger' not in coll:
            sta_coll_names.append(coll)
            list_help.append(coll)

    # select which collection should be used
    list_help.insert(0, 'Create a new collection')
    collection_name = col1.selectbox('Select a existing collection or create a new one:', list_help)
    # also a new collection can be created
    disable_txt_input = True
    pholder = ''
    if collection_name == 'Create a new collection':
        disable_txt_input = False
        pholder = 'Insert new collection name'
    new_collection_name = col2.text_input('', placeholder=pholder, disabled=disable_txt_input, help='Collection name must contain "station"!')

    # create layout depending if a new collection is created or an existing is used
    empty_button = False
    copy_button = False
    existing_button = False
    copy_collection = ''
    if not disable_txt_input:
        # create an empty one or copy an existing
        selected_collection = new_collection_name
        disable_buttons = True
        if new_collection_name != '':
            disable_buttons = False

        col1_new, col2_new = cont.columns([1, 1])
        col1_new.markdown('Create a copy of an existing collection:')
        col2_new.markdown('Create empty collection:')
        empty_button = col2_new.button('CREATE EMPTY COLLECTION', disabled=disable_buttons)
        copy_collection = col1_new.selectbox('Create a copy of the following collection:', sta_coll_names)
        copy_button = col1_new.button('CREATE COPY', disabled=disable_buttons)
    else:
        # use an existing one
        selected_collection = collection_name
        existing_button = cont.button('USE THE COLLECTION')

    # set the variable to change to the station input page
    collection_selected = False
    if empty_button or existing_button or copy_button:
        collection_selected = True

    # create an empty collection
    if empty_button:
        det.create_empty_collection(selected_collection)

    # create a copy of an existing collection
    if copy_button:
        det.clone_colletion_to_colletion(copy_collection, selected_collection)

    return collection_selected, selected_collection


def input_station_information(cont, warning_top, coll_name):
    station_list = ['Station 11 (Nanoq)', 'Station 12 (Terianniaq)', 'Station 13 (Ukaleq)', 'Station 14 (Tuttu)', 'Station 15 (Umimmak)', 'Station 21 (Amaroq)', 'Station 22 (Avinngaq)',
                    'Station 23 (Ukaliatsiaq)', 'Station 24 (Qappik)', 'Station 25 (Aataaq)']
    selected_station = cont.selectbox('Select a station', station_list)
    # get the name and id out of the string
    selected_station_name = selected_station[selected_station.find('(')+1:-1]
    selected_station_id = int(selected_station[len('Station '):len('Station ')+2])

    station_info = load_station_infos(selected_station_id, coll_name)

    # give a warning if the station already exist and is commissioned
    if station_info != {}:
        warning_top.warning(f'A commissioned database entry exits for the selected station: {selected_station}.')

    # enter the station position (if the station already exist, fill in the saved values)
    col1, col2, col3 = cont.columns([1, 1, 1])
    if station_info !={}:
        db_position = station_info[selected_station_id]['position']
    else:
        db_position = [0,0,0]
    x_pos = col1.text_input('x position', value=db_position[0])
    y_pos = col2.text_input('y position', value=db_position[1])
    z_pos = col3.text_input('z position', value=db_position[2])
    position = [float(x_pos), float(y_pos), float(z_pos)]

    # enter the (de)commission time position (if the station already exist, fill in the saved values)
    col11, col22 = cont.columns([1, 1])
    if station_info != {}:
        value_comm = station_info[selected_station_id]['commission_time']
        value_decomm = station_info[selected_station_id]['decommission_time']
    else:
        value_comm = datetime.now()
        value_decomm = datetime(2035, 1, 1)
    comm_date = col11.date_input('commission time', value=value_comm, min_value=datetime(2018, 1, 1), max_value=datetime(2080, 1, 1))
    decomm_date = col22.date_input('decommission time', value=value_decomm, min_value=datetime(2018, 1, 1), max_value=datetime(2100, 1, 1))

    return selected_station_name, selected_station_id, position, comm_date, decomm_date


def build_individual_container(cont):
    col1_iglu, col2_iglu = cont.columns([1, 1])
    iglu_name = col1_iglu.selectbox('IGLU name:', ['Choose a name', 'test'])
    iglu_weight = col2_iglu.number_input('weight:', value=1.0, key='IGLU')
    col1_cable, col2_cable = cont.columns([1, 1])
    cable_name = col1_cable.selectbox('cable name:', ['Choose a name', 'test'])
    cable_weight = col2_cable.number_input('weight:', value=1.0, key='CABLE')
    col1_drab, col2_drab = cont.columns([1, 1])
    drab_name = col1_drab.selectbox('DRAB name:', ['Choose a name', 'test'])
    drab_weigth = col2_drab.number_input('weight:', value=1.0, key='DRAB')

    return [iglu_name, iglu_weight, cable_name, cable_weight, drab_name, drab_weigth]


def build_complete_container(cont):
    downhole_name = cont.selectbox('downhole chain name:', ['Choose a name', 'test'])

    return [downhole_name]


def build_both_container(cont):
    col1_iglu, col2_iglu = cont.columns([1, 1])
    iglu_name = col1_iglu.selectbox('IGLU name:', ['Choose a name', 'test'])
    iglu_weight = col2_iglu.number_input('weight:', value=1.0, key='IGLU')
    col1_cable, col2_cable = cont.columns([1, 1])
    cable_name = col1_cable.selectbox('cable name:', ['Choose a name', 'test'])
    cable_weight = col2_cable.number_input('weight:', value=1.0, key='CABLE')
    col1_drab, col2_drab = cont.columns([1, 1])
    drab_name = col1_drab.selectbox('DRAB name:', ['Choose a name', 'test'])
    drab_weigth = col2_drab.number_input('weight:', value=1.0, key='DRAB')
    downhole_name = cont.selectbox('downhole chain name:', ['Choose a name', 'test'])

    return [iglu_name, iglu_weight, cable_name, cable_weight, drab_name, drab_weigth, downhole_name]


def input_channel_information(cont, station_id, coll_name):
    station_info = load_station_infos(station_id, coll_name)

    # load all channels which are already in the database
    if station_info != {}:
        channels_db = list(station_info[station_id]['channels'].keys())
    else:
        channels_db = []
    cont.info(f'Channels included in the database: {len(channels_db)}/24')
    cont.markdown('General information:')
    col11, col22, col33 = cont.columns([0.5, 0.5, 1])

    # if not all channels are in the db, add the possibility to add another channel number
    channel_help = channels_db
    if len(channels_db) < 24:
        channel_help.insert(0,'new channel number')
        disable_new_entry = False
    else:
        disable_new_entry = True
    channel = col11.selectbox('Select a channel or enter a new channel number:', channel_help, help='The channel number must be an integer between 0 and 23.')
    new_cha = col22.text_input('', placeholder='channel number', disabled=disable_new_entry)
    if channel == 'new channel number':
        selected_channel = new_cha
    else:
        selected_channel = channel

    # if the channel already exist in the database, the channel info will be loaded
    if channel == 'new channel number':
        channel_info = {}
    else:
        channel_info = station_info[station_id]['channels'][selected_channel]


    # if the channel exist
    antenna_names = ['Phased Array Vpol', 'Power String Hpol', "Power String Vpol", "Helper String B Vpol", "Helper String B Hpol", 'rno_surface, Power String', 'rno_surface, Helper B', 'rno_surface, Helper C',
                     'Helper String C Vpol', 'Helper String C Hpol']
    selected_antenna_name = col33.selectbox('Select a antenna name:', antenna_names)

    col111, col222, col333 = cont.columns([1, 1, 1])
    x_pos_ant = col111.text_input('x position', key='x_antenna')
    y_pos_ant = col222.text_input('y position', key='y_antenna')
    z_pos_ant = col333.text_input('z position', key='z_antenna')
    position = [x_pos_ant, y_pos_ant, z_pos_ant]

    col1a, col2a, col3a, col4a = cont.columns([1, 1, 1, 1])
    ant_ori_theta = col1a.text_input('orientation (theta):')
    ant_ori_phi = col2a.text_input('orientation (phi):')
    ant_rot_theta = col3a.text_input('rotation (theta):')
    ant_rot_phi = col4a.text_input('rotation (phi):')
    ori_rot = [ant_ori_theta, ant_ori_phi, ant_rot_theta, ant_rot_phi]

    diff_antenna_types = ["RNOG_vpol_4inch_center_1.73", "RNOG_quadslot_v3_air_rescaled_to_n1.74", "createLPDA_100MHz_InfFirn_n1.4"]
    selected_antenna_type = cont.selectbox('Select antenna type:', diff_antenna_types)

    coltime1, coltime2 = cont.columns([1, 1])
    comm_date_ant = coltime1.date_input('commission time', min_value=datetime(2018, 1, 1), max_value=datetime(2080, 1, 1), key='comm_time_antenna')
    decomm_date_ant = coltime2.date_input('decommission time', value=datetime(2035, 1, 1), min_value=datetime(2018, 1, 1), max_value=datetime(2100, 1, 1), key='decomm_time_antenna')

    cont.markdown('Signal chain:')
    type_signal_chain = cont.radio('Input form of the signal chain:', ['individual components', 'complete chain', 'both'], horizontal=True)
    signal_chain_cont = cont.container()
    signal_chain = []
    if type_signal_chain == 'individual components':
        signal_chain_cont.empty()
        signal_chain = build_individual_container(signal_chain_cont)
    elif type_signal_chain == 'complete chain':
        signal_chain_cont.empty()
        signal_chain = build_complete_container(signal_chain_cont)
    elif type_signal_chain == 'both':
        signal_chain_cont.empty()
        signal_chain = build_both_container(signal_chain_cont)

    return selected_channel, selected_antenna_name, position, ori_rot, selected_antenna_type, comm_date_ant, decomm_date_ant, signal_chain


def test(station_id):
    det.update(datetime.now())
    station_info = det.get_station_information(station_id)
    print(station_info)
    # print(station_info[station_id]['name'])
    # print(station_info[station_id]['channels'].keys())


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
        if primary and antenna_name in det.get_object_names(page_name):
            det.update_primary(page_name, antenna_name)
        det.set_not_working(page_name, antenna_name, primary)
    else:
        if primary and antenna_name in det.get_object_names(page_name):
            det.update_primary(page_name, antenna_name)
        det.antenna_add_Sparameter(page_name, antenna_name, [s_name], data, primary, protocol, input_units)
