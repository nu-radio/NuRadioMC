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
# det = Detector(config.DATABASE_TARGET)
# det = Detector(database_connection='test')


def load_general_station_infos(station_id, coll_name):
    det.update(datetime.now(), coll_name)
    station_info = det.get_general_station_information(coll_name, station_id)
    return station_info


def load_object_names(obj_type):
    return det.get_object_names(obj_type)


def load_measurement_names(collection):
    return det.get_quantity_names(collection, 'measurements.measurement_name')


def load_current_primary_device_information(station, device_id):
    """function to load the information about the specified device which is currently primary"""
    return det.get_complete_device_information(station, device_id)


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

    # create layout depending on if a new collection is created or an existing is used
    empty_button = False
    copy_button = False
    existing_button = False
    copy_collection = ''
    if not disable_txt_input:
        # create an empty one or copy an existing
        selected_collection = new_collection_name
        disable_buttons = True
        if new_collection_name != '' and 'station' in new_collection_name:
            disable_buttons = False

        col1_new, col2_new = cont.columns([1, 1])
        col1_new.markdown('Create a copy of an existing collection:')
        col2_new.markdown('Create empty collection:')
        empty_button = col2_new.button('CREATE EMPTY COLLECTION', disabled=disable_buttons)
        copy_collection = col1_new.selectbox('Create a copy of the following collection:', sta_coll_names, label_visibility='collapsed')
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

    station_info = load_general_station_infos(selected_station_id, coll_name)

    # give a warning if the station already exist and is commissioned
    if station_info != {}:
        warning_top.warning(f'A commissioned database entry exits for the selected station: {selected_station}.')

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

    # the commission and decommission date are only give as dates, but the input should be date + time
    comm_date = datetime.combine(comm_date, time(0, 0, 0))
    decomm_date = datetime.combine(decomm_date, time(0, 0, 0))

    # plain text input -> a general comment about the station can be given to be saved in the database (e.g.: Wind turbine)
    if station_info != {} and 'station_comment' in station_info[selected_station_id].keys():
        initial_comment = station_info[selected_station_id]['station_comment']
    else:
        initial_comment = ''
    comment = cont.text_area('General comments about the station:', value=initial_comment)

    return selected_station_name, selected_station_id, comm_date, decomm_date, comment, station_info


def build_individual_container(cont, channel):
    cont_warning = cont.container()
    if channel in [12,13,14,15,16,17,18,19,20]:
        # initialize the surface board name key -> needed to make the surface board component selection possible
        if 'surface_board_name_key' not in st.session_state:
            st.session_state['surface_board_name_key'] = 'Choose a name'

        if st.session_state.surface_board_name_key == 'Choose a name':
            search_surface_board_name = None
            disable_surface_board_components = True
        else:
            search_surface_board_name = st.session_state.surface_board_name_key
            disable_surface_board_components = False

        # get the surface board and surface cable names from the database:
        db_surf_board_names = det.get_object_names('surface_board')
        db_surf_cable_names = det.get_object_names('surface_cable')

        # get all channel-id of the surface board from the database (depends on the selected surface board name)
        db_surface_board_info = det.get_all_available_signal_chain_configs(collection='surface_board', object_name=search_surface_board_name, input_dic={'channel_id': None})
        db_surface_board_channels = db_surface_board_info['channel_id']

        # insert the standard display option
        db_surf_board_names.insert(0, 'Choose a name')
        db_surf_cable_names.insert(0, 'Choose a name')
        db_surface_board_channels.insert(0, 'Select an option')

        # create the streamlit page
        col_surf, col_cable = cont.columns([1,1])
        col_surf.markdown('**surface board**')
        surface_board_name = col_surf.selectbox('surface board name:', db_surf_board_names, label_visibility='collapsed', key='surface_board_name_key')
        surface_board_channel = col_surf.selectbox('channel-id: ', db_surface_board_channels, disabled=disable_surface_board_components)
        col_cable.markdown('**surface cable**')
        surface_cable_name = col_cable.selectbox('surface cable name:', db_surf_cable_names, label_visibility='collapsed')

        return {'surface_board': surface_board_name, 'surface_cable': surface_cable_name, 'surface_board_channel_id': surface_board_channel}
    else:
        # initialize the drab name key -> needed to make the drab component selection possible
        if 'drab_name_key' not in st.session_state:
            st.session_state['drab_name_key'] = 'Choose a name'

        if st.session_state.drab_name_key == 'Choose a name':
            search_drab_name = None
            disable_drab_components = True
        else:
            search_drab_name = st.session_state.drab_name_key
            disable_drab_components = False

        # get the iglu, drab and cable names from the database
        db_iglu_board_names = det.get_object_names('iglu_board')
        db_drab_board_names = det.get_object_names('drab_board')
        db_down_cable_names = det.get_object_names('downhole_cable')

        # get all channel-id of the drab from the database (depends on the selected drab name)
        db_drab_info = det.get_all_available_signal_chain_configs(collection='drab_board', object_name=search_drab_name, input_dic={'channel_id': None})
        db_drab_channels = db_drab_info['channel_id']

        # insert the standard display option
        db_iglu_board_names.insert(0, 'Choose a name')
        db_drab_board_names.insert(0, 'Choose a name')
        db_down_cable_names.insert(0, 'Choose a name')
        db_drab_channels.insert(0, 'Select an option')

        # create the streamlit page
        col_iglu, col_cable, col_drab = cont.columns([1,1,1])
        col_iglu.markdown('**IGLU**')
        iglu_name = col_iglu.selectbox('name:', db_iglu_board_names, key='iglu_name_key', label_visibility='collapsed')
        col_cable.markdown('**downhole cable**')
        cable_name = col_cable.selectbox('name:', db_down_cable_names, label_visibility='collapsed')
        col_drab.markdown('**DRAB**')
        drab_name = col_drab.selectbox('name:', db_drab_board_names, key='drab_name_key', label_visibility='collapsed')
        drab_channel = col_drab.selectbox('channel-id:', db_drab_channels, disabled=disable_drab_components)

        return {'iglu_board': iglu_name, 'downhole_cable': cable_name, 'drab_board': drab_name, 'drab_board_channel_id': drab_channel}


def build_complete_container(cont, channel):
    cont_warning = cont.container()
    if channel in [12, 13, 14, 15, 16, 17, 18, 19, 20]:
        surface_chain = cont.selectbox('surface chain: Not existing yet', [])
        return {'surface_chain': 'not existing yet'}
    else:
        # initialize the downhole_chain name key -> needed to make the downhole_chain component selection possible
        if 'downhole_chain_name_key' not in st.session_state:
            st.session_state['downhole_chain_name_key'] = 'Choose a name'
        if 'downhole_chain_breakout_key' not in st.session_state:
            st.session_state['downhole_chain_breakout_key'] = 'Select an option'
        if 'downhole_chain_breakout_channel_key' not in st.session_state:
            st.session_state['downhole_chain_breakout_channel_key'] = 'Select an option'

        if st.session_state.downhole_chain_name_key == 'Choose a name':
            search_downhole_chain_name = None
            disable_downhole_chain_components = True
        else:
            search_downhole_chain_name = st.session_state.downhole_chain_name_key
            disable_downhole_chain_components = False
        if st.session_state.downhole_chain_breakout_key == 'Select an option':
            search_downhole_breakout = None
        else:
            search_downhole_breakout = st.session_state.downhole_chain_breakout_key
        if st.session_state.downhole_chain_breakout_channel_key == 'Select an option':
            search_downhole_breakout_channel = None
        else:
            search_downhole_breakout_channel = st.session_state.downhole_chain_breakout_channel_key

        # get all possible downhole chain names from the database
        db_down_chain_name = det.get_object_names('downhole_chain')
        db_down_chain_name.insert(0, 'Choose a name')

        # get all channel-id of the drab from the database (depends on the selected drab name)
        db_downhole_chain_info = det.get_all_available_signal_chain_configs(collection='downhole_chain', object_name=search_downhole_chain_name, input_dic={'breakout': search_downhole_breakout, 'breakout_channel': search_downhole_breakout_channel})
        db_downhole_chain_breakout_ids = db_downhole_chain_info['breakout']
        db_downhole_chain_breakout_channel_ids = db_downhole_chain_info['breakout_channel']
        db_downhole_chain_breakout_ids.insert(0, 'Select an option')
        db_downhole_chain_breakout_channel_ids.insert(0, 'Select an option')

        downhole_name = cont.selectbox('downhole chain name:', db_down_chain_name, key='downhole_chain_name_key')
        downhole_breakout_id = cont.selectbox('downhole chain breakout id:', db_downhole_chain_breakout_ids, key='downhole_chain_breakout_key', disabled=disable_downhole_chain_components)
        downhole_breakout_channel_id = cont.selectbox('downhole chain breakout channel id:', db_downhole_chain_breakout_channel_ids, key='downhole_chain_breakout_channel_key', disabled=disable_downhole_chain_components)

        return {'downhole_chain': downhole_name, 'downhole_chain_breakout': downhole_breakout_id, 'downhole_chain_breakout_channel': downhole_breakout_channel_id}


def input_channel_information(cont, station_id, coll_name, station_info):
    # load all channels which are already in the database
    if station_info != {}:
        channels_db = []
        for key in station_info[station_id]['channels'].keys():
            channels_db.append(station_info[station_id]['channels'][key]['id'])
        # channels_db = list(station_info[station_id]['channels'].keys())
    else:
        channels_db = []
    cont.info(f'Channels included in the database: {len(channels_db)}/24')
    cont.markdown('General information:')
    col11, col22, col33 = cont.columns([0.5, 0.5, 1])

    # if not all channels are in the db, add the possibility to add another channel number
    channel_help = channels_db
    if len(channels_db) < 24:
        channel_help.insert(0, 'new channel number')
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

    # tranform the channel number from a string into an int
    if selected_channel != '':
        selected_channel = int(selected_channel)

    # if the channel exist make the existing antenna name the default argument, else display the names as listed here
    antenna_names = ['Phased Array Vpol', 'Power String Hpol', "Power String Vpol", "Helper String B Vpol", "Helper String B Hpol", 'rno_surface, Power String', 'rno_surface, Helper B', 'rno_surface, Helper C',
                     'Helper String C Vpol', 'Helper String C Hpol']
    if channel_info != {}:
        db_ant_name = channel_info['ant_name']
        db_ant_name_index = np.where(np.asarray(antenna_names) == db_ant_name)[0][0]

        # add the existing antenna name to the front of the list
        antenna_names.pop(db_ant_name_index)
        antenna_names.insert(0, db_ant_name)

    selected_antenna_name = col33.selectbox('Select a antenna name:', antenna_names)

    # select the antenna type; if the channel exists, the antenna type of the db will be used as default otherwise the antenna types as shown in the list will be given
    diff_antenna_types = ["RNOG_vpol_4inch_center_1.73", "RNOG_quadslot_v3_air_rescaled_to_n1.74", "createLPDA_100MHz_InfFirn_n1.4"]
    if channel_info != {}:
        db_ant_type = channel_info['type']
        db_type_index = np.where(np.asarray(diff_antenna_types) == db_ant_type)[0][0]
        # isert the antenna type as the defalut
        diff_antenna_types.pop(db_type_index)
        diff_antenna_types.insert(0, db_ant_type)
    selected_antenna_type = cont.selectbox('Select antenna type:', diff_antenna_types)

    # input the (de)commisschon time of the channel (can be used to set a channel to be broken); if the channel exist the dates from the db will be used as a default
    coltime1, coltime2 = cont.columns([1, 1])
    if channel_info != {}:
        comm_starting_date = channel_info['commission_time']
        decomm_starting_date = channel_info['decommission_time']
    else:
        comm_starting_date = datetime.now()
        decomm_starting_date = datetime(2080, 1, 1)
    comm_date_ant = coltime1.date_input('commission time', value=comm_starting_date, min_value=datetime(2018, 1, 1), max_value=datetime(2080, 1, 1), key='comm_time_antenna')
    decomm_date_ant = coltime2.date_input('decommission time', value=decomm_starting_date, min_value=datetime(2018, 1, 1), max_value=datetime(2100, 1, 1), key='decomm_time_antenna')
    # the commission and decommission date are only give as dates, but the input should be date + time
    comm_date_ant = datetime.combine(comm_date_ant, time(0, 0, 0))
    decomm_date_ant = datetime.combine(decomm_date_ant, time(0, 0, 0))

    # input if a channel is broken
    cont.markdown('Is the channel working normally?')
    function_channel = cont.checkbox('Channel is working', value=True)

    # input the signal chain; if the channel already exists the entries from the db will be used as the default
    cont.markdown('Signal chain:')

    # get the signal chain from the db
    # if channel_info != {}:
    #     db_signal_chain = channel_info['signal_ch']
    # else:
    #     db_signal_chain = []

    signal_chain_cont = cont.container()
    signal_chain = []
    signal_chain_cont.empty()
    signal_chain = build_individual_container(signal_chain_cont, selected_channel)

    # plain text input -> a comment about the channel can be given to be saved in the database
    cont.markdown('Comments:')
    if station_info != {} and 'channel_comment' in station_info[station_id].keys():
        initial_comment = station_info[station_id]['channel_comment']
    else:
        initial_comment = ''
    comment = cont.text_area('Comment about the channel performance:', value=initial_comment, label_visibility='collapsed')

    return selected_channel, selected_antenna_name, selected_antenna_type, comm_date_ant, decomm_date_ant, signal_chain, comment, function_channel


def input_device_information(cont, station_id, station_info):
    # load all devices which are already in the database
    print(station_info)
    if station_info != {}:
        devices_db = list(station_info[station_id]['devices'].keys())
    else:
        devices_db = []
    cont.info(f'Number of devices stored in the database: {len(devices_db)}')

    cont.markdown('General information:')
    col_d1, col_d2, col_d3 = cont.columns([0.5, 0.5, 1])

    devices_help = devices_db
    devices_help.insert(0, 'new device id')
    device_id = col_d1.selectbox('Select a device id or enter a new device id:', devices_help, help='The channel number must be an integer between 0 and 23.')
    disabled_id_input = True
    if device_id == 'new device id':
        disabled_id_input = False
    new_device_id = col_d2.text_input('', disabled=disabled_id_input, placeholder='device id')
    if device_id == 'new device id':
        selected_device_id = new_device_id
    else:
        selected_device_id = device_id

    # if the device already exist in the database, the device info will be loaded
    if device_id == 'new device id':
        device_info = {}
    else:
        device_info = station_info[station_id]['devices'][selected_device_id]

    # tranform the device id from a string into an int
    if selected_device_id != '':
        selected_device_id = int(selected_device_id)

    # if the device exist make the existing device name the default argument, else display the names as listed here
    device_names = ['Helper String B CAL Vpol', 'Helper String C CAL Vpol', 'Surface CAL Vpol', 'Solar panel 1', 'Solar panel 1', 'wind-turbine', 'daq box']
    if device_info != {}:
        db_device_name = device_info['device_name']
        db_device_name_index = np.where(np.asarray(device_names) == db_device_name)[0][0]

        # add the existing antenna name to the front of the list
        device_names.pop(db_device_name_index)
        device_names.insert(0, db_device_name)

    selected_device_name = col_d3.selectbox('Select a device name:', device_names)

    # input the (de)commisschon time of the device; if the device exist the dates from the db will be used as a default
    col_d_time1, col_d_time2 = cont.columns([1, 1])
    if device_info != {}:
        dev_comm_starting_date = device_info['commission_time']
        dev_decomm_starting_date = device_info['decommission_time']
    else:
        dev_comm_starting_date = datetime.now()
        dev_decomm_starting_date = datetime(2080, 1, 1)
    dev_comm_date = col_d_time1.date_input('commission time', value=dev_comm_starting_date, min_value=datetime(2018, 1, 1), max_value=datetime(2080, 1, 1), key='dev_comm_time')
    dev_decomm_date = col_d_time2.date_input('decommission time', value=dev_decomm_starting_date, min_value=datetime(2018, 1, 1), max_value=datetime(2100, 1, 1), key='dev_decomm_time')
    # the commission and decommission date are only give as dates, but the input should be date + time
    dev_comm_date = datetime.combine(dev_comm_date, time(0, 0, 0))
    dev_decomm_date = datetime.combine(dev_decomm_date, time(0, 0, 0))

    # input if a device is broken
    cont.markdown('Is the device working normally?')
    function_device = cont.checkbox('Device is working', value=True)

    # select the amplifier (IGLU)
    # only show this if a pulser is selected
    iglu_db = det.get_object_names('iglu_board')
    if 'CAL' in selected_device_name:
        selected_amp = cont.selectbox('Select an IGLU:', iglu_db)
    else:
        selected_amp = None

    # plain text input -> a comment about the device can be given to be saved in the database
    cont.markdown('Comments:')
    if station_info != {} and 'device_comment' in station_info[station_id].keys():
        initial_comment = station_info[station_id]['device_comment']
    else:
        initial_comment = ''
    comment = cont.text_area('Comment about the device performance:', value=initial_comment, label_visibility='collapsed')

    return selected_device_id, selected_device_name, dev_comm_date, dev_decomm_date, comment, function_device, selected_amp


def validate_station_inputs(container_bottom, comm_date_station, decomm_date_station):
    dates_station_correct = False

    disable_insert_button = True

    # validate that decomm_date > comm_date
    if decomm_date_station > comm_date_station:
        dates_station_correct = True
    else:
        container_bottom.error('The decommission date of the station must be later than the commission date.')

    if dates_station_correct:
        disable_insert_button = False

    return disable_insert_button


def validate_channel_inputs(collection, container_bottom, station_name, comm_date_channel, deomm_date_channel, channel, signal_chain):
    dates_channel_correct = False
    channel_correct = False
    signal_chain_correct = False
    station_in_db = False

    disable_insert_button = True
    # validate that decomm_date > comm_date
    if deomm_date_channel > comm_date_channel:
        dates_channel_correct = True
    else:
        container_bottom.error('The decommission date of the channel must be later than the commission date.')

    # validate that a valid channel is given
    possible_channel_ids = np.arange(0,24,1)
    if channel != '':
        if channel in possible_channel_ids:
            channel_correct = True
        else:
            container_bottom.error('The channel number must be between 0 and 23.')

    # validate that signal chain input is given
    if 'Choose a name' not in signal_chain and 'not existing yet' not in signal_chain:
        signal_chain_correct = True
    else:
        container_bottom.error('Not all options for the signal chain are filled.')

    # check if there is an entry for the station in the db
    if station_name in det.get_object_names(collection):
        station_in_db = True
    else:
        container_bottom.error('There is no corresponding entry for the station in the database.')

    if dates_channel_correct and channel_correct and signal_chain_correct and station_in_db:
        disable_insert_button = False

    return disable_insert_button


def validate_device_inputs(container_bottom, station_name, comm_date, deomm_date, device_id):
    dates_correct = False
    device_id_correct = False
    station_in_db = False

    disable_insert_button = True

    # validate that decomm_date > comm_date
    if deomm_date > comm_date:
        dates_correct = True
    else:
        container_bottom.error('The decommission date of the device must be later than the commission date.')

    # validate that a valid channel is given
    if device_id != '':
        if device_id >= 0:
            device_id_correct = True
        else:
            container_bottom.error('The device id must be larger than 0.')
    else:
        container_bottom.error('Please select or enter a device id.')

    # check if there is an entry for the station in the db
    if station_name in det.get_object_names('station_rnog'):
        station_in_db = True
    else:
        container_bottom.error('There is no corresponding entry for the station in the database.')

    if dates_correct and device_id_correct and station_in_db:
        disable_insert_button = False

    return disable_insert_button


def insert_general_station_info_to_db(station_id, collection_name, station_name, station_comment, station_comm_time, station_decomm_time):
    det.add_general_station_info(collection_name, station_id, station_name, station_comment, station_comm_time, station_decomm_time)


def insert_general_channel_info_to_db(station_id, collection_name, channel_id, signal_chain, ant_name, channel_type, channel_comment, commission_time, decommission_time):
    # convert the signal chain to the correct format
    # converted_signal_chain = []
    # # for i in range(int(len(signal_chain)/2)):
    # #     print(signal_chain)
    # #     converted_signal_chain.append({'type': signal_chain[2*i], 'uname': signal_chain[2*i + 1]})
    # print(signal_chain)
    # for key in signal_chain:
    #     converted_signal_chain.append({'type': key, 'uname': signal_chain[key]})
    #
    # # the check if the channel already exists happens in add_channel_to_station
    # det.add_general_channel_info_to_station(collection_name, station_id, channel_id, converted_signal_chain, ant_name, channel_type, channel_comment, commission_time, decommission_time)
    det.add_general_channel_info_to_station(collection_name, station_id, channel_id, signal_chain, ant_name, channel_type, channel_comment, commission_time, decommission_time)


def insert_general_device_info_to_db(station_id, collection_name, device_id, device_name, amp_name, device_comment, commission_time, decommission_time):
    det.add_general_device_info_to_station(collection_name, station_id, device_id, device_name, device_comment, amp_name, commission_time, decommission_time)


def insert_channel_position_to_db(station_id, channel_id, measurement_name, measurement_time, position, orientation, rotation, primary):
    det.add_channel_position(station_id, channel_id, measurement_name, measurement_time, position, orientation, rotation, primary)


def insert_station_position_to_db(station_id, measurement_name, measurement_time, position, primary):
    det.add_station_position(station_id, measurement_name, measurement_time, position, primary)


def insert_signal_chain_to_db(station_id, channel_number, config_name, sig_chain, primary, primary_components):
    det.add_channel_signal_chain(station_id, channel_number, config_name, sig_chain, primary, primary_components)


def insert_device_position_to_db(station_id, device_id, measurement_name, measurement_time, position, orientation, rotation, primary):
    det.add_device_position(station_id, device_id, measurement_name, measurement_time, position, orientation, rotation, primary)