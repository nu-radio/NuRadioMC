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

    # the commission and decommission date are only give as dates, but the input should be date + time
    comm_date = datetime.combine(comm_date, time(0, 0, 0))
    decomm_date = datetime.combine(decomm_date, time(0, 0, 0))

    # plain text input -> a general comment about the station can be given to be saved in the database (e.g.: Wind turbine)
    if station_info != {} and 'station_comment' in station_info[selected_station_id].keys():
        initial_comment = station_info[selected_station_id]['station_comment']
    else:
        initial_comment = ''
    comment = cont.text_area('General comments about the station:', value=initial_comment)

    return selected_station_name, selected_station_id, position, comm_date, decomm_date, comment


def build_individual_container(cont, channel, db_signal_chain):
    cont_warning = cont.container()
    if channel in [12,13,14,15,16,17,18,19,20]:
        # get all possible surface board names from the data base:
        db_surf_board_names = det.get_object_names('surface_board')
        # get all possible surface cable names from the db:
        db_surf_cable_names = det.get_object_names('surface_cable')
        # if channel exist, put the information of the existing channel as the default
        if db_signal_chain != []:
            db_cha_surf_board = db_signal_chain[0]['uname']
            db_cha_surf_board_weight = db_signal_chain[0]['weight']
            db_cha_surf_cable = db_signal_chain[1]['uname']
            db_cha_surf_cable_weight = db_signal_chain[1]['weight']

            # put the cable and board names at the front of the name lists
            if db_cha_surf_board in db_surf_board_names:
                db_cha_surf_board_index = np.where(np.asarray(db_surf_board_names) == db_cha_surf_board)[0][0]
                db_surf_board_names.pop(db_cha_surf_board_index)
            else:
                cont_warning.warning('The chosen SURFACE BOARD is not part of the database!')
            db_surf_board_names.insert(0, db_cha_surf_board)

            if db_cha_surf_cable in db_surf_cable_names:
                db_cha_surf_cable_index = np.where(np.asarray(db_surf_cable_names) == db_cha_surf_cable)[0][0]
                db_surf_cable_names.pop(db_cha_surf_cable_index)
            else:
                cont_warning.warning('The chosen SURFACE CABLE is not part of the database!')
            db_surf_cable_names.insert(0, db_cha_surf_cable)
        else:
            db_surf_board_names.insert(0, 'Choose a name')
            db_surf_cable_names.insert(0, 'Choose a name')
            db_cha_surf_board_weight = 1
            db_cha_surf_cable_weight = 1

        col1_surf, col2_surf = cont.columns([1, 1])
        surface_board_name = col1_surf.selectbox('surface board name:', db_surf_board_names)
        surface_board_weight = col2_surf.number_input('weight:', value=db_cha_surf_board_weight, key='SURFACE_board')
        col1_cable, col2_cable = cont.columns([1, 1])
        surface_cable_name = col1_cable.selectbox('surface cable name:', db_surf_cable_names)
        surface_cable_weight = col2_cable.number_input('weight:', value=db_cha_surf_cable_weight, key='CABLE')

        return ['surface_board', surface_board_name, surface_board_weight, 'surface_cable', surface_cable_name, surface_cable_weight]
    else:
        # get all possible iglu board names from the database:
        db_iglu_board_names = det.get_object_names('iglu_board')
        # get all possible drab board names from the database:
        db_drab_board_names = det.get_object_names('drab_board')
        # get all possible downhole cable names from the db:
        db_down_cable_names = det.get_object_names('downhole_cable')
        # if channel exist, put the information of the existing channel as the default
        if db_signal_chain != []:
            db_cha_iglu = db_signal_chain[0]['uname']
            db_cha_iglu_weight = db_signal_chain[0]['weight']
            db_cha_drab = db_signal_chain[2]['uname']
            db_cha_drab_weight = db_signal_chain[2]['weight']
            db_cha_down_cable = db_signal_chain[1]['uname']
            db_cha_down_cable_weight = db_signal_chain[1]['weight']

            # put the cable and board names at the front of the name lists
            if db_cha_iglu in db_iglu_board_names:
                db_cha_iglu_index = np.where(np.asarray(db_iglu_board_names) == db_cha_iglu)[0][0]
                db_iglu_board_names.pop(db_cha_iglu_index)
            else:
                cont_warning.warning('The chosen IGLU BOARD is not part of the database!')
            db_iglu_board_names.insert(0, db_cha_iglu)

            if db_cha_drab in db_drab_board_names:
                db_cha_drab_index = np.where(np.asarray(db_drab_board_names) == db_cha_drab)[0][0]
                db_drab_board_names.pop(db_cha_drab_index)
            else:
                cont_warning.warning('The chosen DRAB BOARD is not part of the database!')
            db_drab_board_names.insert(0, db_cha_drab)

            if db_cha_down_cable in db_down_cable_names:
                db_cha_down_cable_index = np.where(np.asarray(db_down_cable_names) == db_cha_down_cable)[0][0]
                db_down_cable_names.pop(db_cha_down_cable_index)
            else:
                cont_warning.warning('The chosen DOWNHOLE CABLE is not part of the database!')
            db_down_cable_names.insert(0, db_cha_down_cable)
        else:
            db_iglu_board_names.insert(0, 'Choose a name')
            db_drab_board_names.insert(0, 'Choose a name')
            db_down_cable_names.insert(0, 'Choose a name')
            db_cha_iglu_weight = 1
            db_cha_down_cable_weight = 1
            db_cha_drab_weight = 1

        col1_iglu, col2_iglu = cont.columns([1, 1])
        iglu_name = col1_iglu.selectbox('iglu name:', db_iglu_board_names)
        iglu_weight = col2_iglu.number_input('weight:', value=db_cha_iglu_weight, key='IGLU')
        col1_cable, col2_cable = cont.columns([1, 1])
        cable_name = col1_cable.selectbox('downhole cable name:', db_down_cable_names)
        cable_weight = col2_cable.number_input('weight:', value=db_cha_down_cable_weight, key='CABLE')
        col1_drab, col2_drab = cont.columns([1, 1])
        drab_name = col1_drab.selectbox('drab name:', db_drab_board_names)
        drab_weigth = col2_drab.number_input('weight:', value=db_cha_drab_weight, key='DRAB')

        return ['iglu_board', iglu_name, iglu_weight, 'downhole_cable', cable_name, cable_weight, 'drab_board', drab_name, drab_weigth]


def build_complete_container(cont, channel, db_signal_chain):
    cont_warning = cont.container()
    if channel in [12, 13, 14, 15, 16, 17, 18, 19, 20]:
        surface_chain = cont.selectbox('surface chain: Not existing yet', [])
        return ['surface_chain', 'not existing yet']
    else:
        # get all possible downhole chain names from the database
        db_down_chain_name = det.get_object_names('downhole_chain')
        if db_signal_chain != []:
            if len(db_signal_chain) == 1:
                db_cha_down = db_signal_chain[0]['uname']
                # put the cable and board names at the front of the name lists
                if db_cha_down in db_down_chain_name:
                    db_cha_down_index = np.where(np.asarray(db_down_chain_name) == db_cha_down)[0][0]
                    db_down_chain_name.pop(db_cha_down_index)
                else:
                    cont_warning.warning('The chosen DOWNHOLE CHAIN is not part of the database!')
                db_down_chain_name.insert(0, db_cha_down)
            else:
                db_down_chain_name.insert(0, 'Choose a name')
        else:
            db_down_chain_name.insert(0, 'Choose a name')

        downhole_name = cont.selectbox('downhole chain name:', db_down_chain_name)
        return ['downhole_chain', downhole_name]


def build_both_container(cont, channel, db_signal_chain):
    cont_warning = cont.container()
    if channel in [12, 13, 14, 15, 16, 17, 18, 19, 20]:
        # get all possible surface board names from the data base:
        db_surf_board_names = det.get_object_names('surface_board')
        # get all possible surface cable names from the db:
        db_surf_cable_names = det.get_object_names('surface_cable')
        # if channel exist, put the information of the existing channel as the default
        if db_signal_chain != []:
            db_cha_surf_board = db_signal_chain[0]['uname']
            db_cha_surf_board_weight = db_signal_chain[0]['weight']
            db_cha_surf_cable = db_signal_chain[1]['uname']
            db_cha_surf_cable_weight = db_signal_chain[1]['weight']

            # put the cable and board names at the front of the name lists
            if db_cha_surf_board in db_surf_board_names:
                db_cha_surf_board_index = np.where(np.asarray(db_surf_board_names) == db_cha_surf_board)[0][0]
                db_surf_board_names.pop(db_cha_surf_board_index)
            else:
                cont_warning.warning('The chosen SURFACE BOARD is not part of the database!')
            db_surf_board_names.insert(0, db_cha_surf_board)

            if db_cha_surf_cable in db_surf_cable_names:
                db_cha_surf_cable_index = np.where(np.asarray(db_surf_cable_names) == db_cha_surf_cable)[0][0]
                db_surf_cable_names.pop(db_cha_surf_cable_index)
            else:
                cont_warning.warning('The chosen SURFACE CABLE is not part of the database!')
            db_surf_cable_names.insert(0, db_cha_surf_cable)
        else:
            db_surf_board_names.insert(0, 'Choose a name')
            db_surf_cable_names.insert(0, 'Choose a name')
            db_cha_surf_board_weight = 1
            db_cha_surf_cable_weight = 1

        col1_surf, col2_surf = cont.columns([1, 1])
        surface_board_name = col1_surf.selectbox('surface board name:', db_surf_board_names)
        surface_board_weight = col2_surf.number_input('weight:', value=db_cha_surf_board_weight, key='SURFACE_board')
        col1_cable, col2_cable = cont.columns([1, 1])
        surface_cable_name = col1_cable.selectbox('surface cable name:', db_surf_cable_names)
        surface_cable_weight = col2_cable.number_input('weight:', value=db_cha_surf_cable_weight, key='CABLE')
        surface_chain = cont.selectbox('surface chain: Not existing yet', [])

        return ['surface_board', surface_board_name, surface_board_weight, 'surface_cable', surface_cable_name, surface_cable_weight, 'surface_chain', 'not existing yet']
    else:
        # get all possible iglu board names from the database:
        db_iglu_board_names = det.get_object_names('iglu_board')
        # get all possible drab board names from the database:
        db_drab_board_names = det.get_object_names('drab_board')
        # get all possible downhole cable names from the db:
        db_down_cable_names = det.get_object_names('downhole_cable')
        # get all possible downhole chain names from the database
        db_down_chain_name = det.get_object_names('downhole_chain')
        # if channel exist, put the information of the existing channel as the default
        if db_signal_chain != []:
            # this if/else allows to display the individual or complete chain without the necessity that both are in the db
            if len(db_signal_chain) == 3:
                db_cha_iglu = db_signal_chain[0]['uname']
                db_cha_iglu_weight = db_signal_chain[0]['weight']
                db_cha_drab = db_signal_chain[2]['uname']
                db_cha_drab_weight = db_signal_chain[2]['weight']
                db_cha_down_cable = db_signal_chain[1]['uname']
                db_cha_down_cable_weight = db_signal_chain[1]['weight']


                # put the cable and board names at the front of the name lists
                if db_cha_iglu in db_iglu_board_names:
                    db_cha_iglu_index = np.where(np.asarray(db_iglu_board_names) == db_cha_iglu)[0][0]
                    db_iglu_board_names.pop(db_cha_iglu_index)
                else:
                    cont_warning.warning('The chosen IGLU BOARD is not part of the database!')
                db_iglu_board_names.insert(0, db_cha_iglu)

                if db_cha_drab in db_drab_board_names:
                    db_cha_drab_index = np.where(np.asarray(db_drab_board_names) == db_cha_drab)[0][0]
                    db_drab_board_names.pop(db_cha_drab_index)
                else:
                    cont_warning.warning('The chosen DRAB BOARD is not part of the database!')
                db_drab_board_names.insert(0, db_cha_drab)

                if db_cha_down_cable in db_down_cable_names:
                    db_cha_down_cable_index = np.where(np.asarray(db_down_cable_names) == db_cha_down_cable)[0][0]
                    db_down_cable_names.pop(db_cha_down_cable_index)
                else:
                    cont_warning.warning('The chosen DOWNHOLE CABLE is not part of the database!')
                db_down_cable_names.insert(0, db_cha_down_cable)
            else:
                db_iglu_board_names.insert(0, 'Choose a name')
                db_drab_board_names.insert(0, 'Choose a name')
                db_down_cable_names.insert(0, 'Choose a name')
                db_cha_iglu_weight = 1
                db_cha_down_cable_weight = 1
                db_cha_drab_weight = 1

            # this if/else allows to display the individual or complete chain without the necessity that both are in the db
            if len(db_signal_chain) == 1:
                db_cha_down = db_signal_chain[3]['uname']
                if db_cha_down in db_down_chain_name:
                    db_cha_down_index = np.where(np.asarray(db_down_chain_name) == db_cha_down)[0][0]
                    db_down_chain_name.pop(db_cha_down_index)
                else:
                    cont_warning.warning('The chosen DOWNHOLE CHAIN is not part of the database!')
                db_down_chain_name.insert(0, db_cha_down)
            else:
                db_down_chain_name.insert(0, 'Choose a name')
        else:
            db_iglu_board_names.insert(0, 'Choose a name')
            db_drab_board_names.insert(0, 'Choose a name')
            db_down_cable_names.insert(0, 'Choose a name')
            db_down_chain_name.insert(0, 'Choose a name')
            db_cha_iglu_weight = 1
            db_cha_down_cable_weight = 1
            db_cha_drab_weight = 1

        col1_iglu, col2_iglu = cont.columns([1, 1])
        iglu_name = col1_iglu.selectbox('iglu name:', db_iglu_board_names)
        iglu_weight = col2_iglu.number_input('weight:', value=db_cha_iglu_weight, key='IGLU')
        col1_cable, col2_cable = cont.columns([1, 1])
        cable_name = col1_cable.selectbox('downhole cable name:', db_down_cable_names)
        cable_weight = col2_cable.number_input('weight:', value=db_cha_down_cable_weight, key='CABLE')
        col1_drab, col2_drab = cont.columns([1, 1])
        drab_name = col1_drab.selectbox('drab name:', db_drab_board_names)
        drab_weigth = col2_drab.number_input('weight:', value=db_cha_drab_weight, key='DRAB')
        downhole_name = cont.selectbox('downhole chain name:', db_down_chain_name)

        return ['iglu_board', iglu_name, iglu_weight, 'downhole_cable', cable_name, cable_weight, 'drab_board', drab_name, drab_weigth, 'downhole_chain', downhole_name]


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

    # insert the position of the antenna; if the channel already exist enter the position of the existing channel
    col111, col222, col333 = cont.columns([1, 1, 1])
    if channel_info != {}:
        db_ant_position = channel_info['ant_position']
    else:
        db_ant_position = [0,0,0]

    x_pos_ant = col111.text_input('x position', key='x_antenna', value=db_ant_position[0])
    y_pos_ant = col222.text_input('y position', key='y_antenna', value=db_ant_position[1])
    z_pos_ant = col333.text_input('z position', key='z_antenna', value=db_ant_position[2])
    position = [x_pos_ant, y_pos_ant, z_pos_ant]

    # input the orientation, rotation of the antenna; if the channel already exist, insert the values from the database
    col1a, col2a, col3a, col4a = cont.columns([1, 1, 1, 1])
    if channel_info != {}:
        db_ant_ori_theta = channel_info['ant_ori_theta']
        db_ant_ori_phi = channel_info['ant_ori_phi']
        db_ant_rot_theta = channel_info['ant_rot_theta']
        db_ant_rot_phi = channel_info['ant_rot_phi']

        db_ori_rot = [db_ant_ori_theta, db_ant_ori_phi, db_ant_rot_theta, db_ant_rot_phi]
    else:
        db_ori_rot = [0,0,0,0]
    ant_ori_theta = col1a.text_input('orientation (theta):', value=db_ori_rot[0])
    ant_ori_phi = col2a.text_input('orientation (phi):', value=db_ori_rot[1])
    ant_rot_theta = col3a.text_input('rotation (theta):', value=db_ori_rot[2])
    ant_rot_phi = col4a.text_input('rotation (phi):', value=db_ori_rot[3])
    ori_rot = [ant_ori_theta, ant_ori_phi, ant_rot_theta, ant_rot_phi]

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
    if channel_info != {}:
        db_signal_chain = channel_info['signal_ch']
        if len(db_signal_chain) == 1:
            radio_index = 1
        elif len(db_signal_chain) == 3 or len(db_signal_chain) == 2:
            radio_index = 0
        else:
            radio_index = 2
    else:
        radio_index = 0
        db_signal_chain = []

    type_signal_chain = cont.radio('Input form of the signal chain:', ['individual components', 'complete chain', 'both'], horizontal=True, index=radio_index)
    signal_chain_cont = cont.container()
    signal_chain = []
    if type_signal_chain == 'individual components':
        signal_chain_cont.empty()
        signal_chain = build_individual_container(signal_chain_cont, selected_channel, db_signal_chain)
    elif type_signal_chain == 'complete chain':
        signal_chain_cont.empty()
        signal_chain = build_complete_container(signal_chain_cont, selected_channel, db_signal_chain)
    elif type_signal_chain == 'both':
        signal_chain_cont.empty()
        signal_chain = build_both_container(signal_chain_cont, selected_channel, db_signal_chain)

    signal_chain.insert(0, type_signal_chain)

    # plain text input -> a comment about the channel can be given to be saved in the database
    cont.markdown('Comments:')
    if station_info != {} and 'channel_comment' in station_info[station_id].keys():
        initial_comment = station_info[station_id]['channel_comment']
    else:
        initial_comment = ''
    comment = cont.text_area('Comment about the channel performance:')

    return selected_channel, selected_antenna_name, position, ori_rot, selected_antenna_type, comm_date_ant, decomm_date_ant, signal_chain, comment, function_channel


def validate_station_inputs(container_bottom, comm_date_station, decomm_date_station):
    dates_station_correct = False

    disable_insert_button = True

    # validate that decomm_date > comm_date
    if decomm_date_station > comm_date_station:
        dates_station_correct = True
    else:
        container_bottom.error('The decommission date of the station must be later than the commissionn date.')

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


def insert_station_to_db(station_id, collection_name, station_name, station_position, station_comment, station_comm_time, station_decomm_time):
    det.add_station(collection_name, station_id, station_name, station_position, station_comment, station_comm_time, station_decomm_time)


def insert_channel_to_db(station_id, collection_name, station_name, station_position, station_comment, station_comm_time, station_decomm_time, channel_id):
    station_info = load_station_infos(station_id, collection_name)
    if station_info != {}:
        db_channels = list(station_info[station_id]['channels'].keys())
    else:
        db_channels = []
    # det.add_station(collection_name, station_id, station_name, station_position, station_comment, station_comm_time, station_decomm_time)
    det.add_channel_to_station(collection_name, station_id, channel_id)

    # if channel already in database
    # get a warning that you override the channel and if you really want to do this (button)
    # 'older' channel will be decommissioned automatically (also give this as an information
    # inform what will be changed
    # get a success information

    # else
    # get a success information

    #
    # if primary and antenna_name in det.get_object_names(page_name):
    #     det.update_primary(page_name, antenna_name)
    # det.antenna_add_Sparameter(page_name, antenna_name, [s_name], data, primary, protocol, input_units)
