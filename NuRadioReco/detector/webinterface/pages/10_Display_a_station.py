import streamlit as st
from NuRadioReco.detector.webinterface.utils.page_config import page_configuration
from NuRadioReco.detector.webinterface.utils.helper_display_station import build_station_selection, load_station_infos
import pandas as pd
from datetime import datetime

def build_main_page(cont):
    cont.title('Display station information')
    cont.subheader('select station')
    # select the station which should be displayed
    selected_collection, selected_station_name, selected_station_id = build_station_selection(cont)

    # makes sure the data is reloaded if collection or station is changed, otherwise the date stays in the cache
    if 'station_id' not in st.session_state:
        st.session_state['station_id'] = 0
    if 'collection_name' not in st.session_state:
        st.session_state['collection_name'] = ' '

    if st.session_state.collection_name != selected_collection:
        st.session_state.collection_name = selected_collection
        st.experimental_memo.clear()
        st.experimental_rerun()

    if st.session_state.station_id != selected_station_id:
        st.session_state.station_id = selected_station_id
        st.experimental_memo.clear()
        st.experimental_rerun()

    # get the station information
    station_info = load_station_infos(selected_station_id, selected_collection)
    cont.subheader('station info')
    # TODO display the station information
    cont.subheader('channel info')

    channel_dic = station_info[selected_station_id]['channels']

    # load and cache the channel information (is only loaded once)
    @st.experimental_memo
    def get_channel_table():
        if 'channels' in list(station_info[selected_station_id].keys()):
            # transform the channel data in a dataframe
            antenna_name = []
            antenna_position = []
            orientation = []
            rotation = []
            antenna_type = []
            commission_time = []
            decommission_time = []
            comments = []
            channel_ids = []
            for cha in channel_dic.keys():
                channel_ids.append(cha)
                antenna_name.append(channel_dic[cha]['ant_name'])
                antenna_position.append([float(channel_dic[cha]['ant_position'][0]), float(channel_dic[cha]['ant_position'][1]), float(channel_dic[cha]['ant_position'][2])])
                orientation.append([float(channel_dic[cha]['ant_ori_theta']), float(channel_dic[cha]['ant_ori_phi'])])
                rotation.append([float(channel_dic[cha]['ant_rot_theta']), float(channel_dic[cha]['ant_rot_phi'])])
                antenna_type.append(channel_dic[cha]['type'])
                commission_time.append(channel_dic[cha]['commission_time'].date())
                decommission_time.append(channel_dic[cha]['decommission_time'].date())
                if 'channel_comment' in channel_dic[cha].keys():
                    comments.append(channel_dic[cha]['channel_comment'])
                else:
                    comments.append('')
            df = pd.DataFrame({'id': channel_ids, 'antenna name': antenna_name, 'antenna type': antenna_type, 'antenna position (x,y,z)': antenna_position, 'orientation (theta, phi)': orientation, 'rotation (theta, phi)': rotation, 'commission': commission_time, 'decommission': decommission_time, 'comment': comments})
        else:
            df = pd.DataFrame({'A': []})

        return df

    # load and cache the signal chain information (is only loaded once)
    #TODO add the complete chain measurement
    @st.experimental_memo
    def get_signal_chain_table():
        if 'channels' in list(station_info[selected_station_id].keys()):
            # transform the channel data in a dataframe
            channel_ids = []
            comp_type = {'comp1': [], 'comp2': [], 'comp3': []}
            comp_name = {'comp1': [], 'comp2': [], 'comp3': []}
            comp_weight = {'comp1': [], 'comp2': [], 'comp3': []}
            for cha in channel_dic.keys():
                channel_ids.append(cha)
                # transform the signal chain data
                signal_chain = channel_dic[cha]['signal_ch']
                # TODO: also consider the case where only the complete chain is in the db
                if len(signal_chain) < 3:
                    for i_sig, sig_dic in enumerate(signal_chain):
                        comp_type[f'comp{i_sig + 1}'].append(sig_dic['type'])
                        comp_name[f'comp{i_sig + 1}'].append(sig_dic['uname'])
                        comp_weight[f'comp{i_sig + 1}'].append(int(sig_dic['weight']))
                    comp_type[f'comp{3}'].append(None)
                    comp_name[f'comp{3}'].append(None)
                    comp_weight[f'comp{3}'].append(None)
                else:
                    for i_sig, sig_dic in enumerate(signal_chain):
                        comp_type[f'comp{i_sig + 1}'].append(sig_dic['type'])
                        comp_name[f'comp{i_sig + 1}'].append(sig_dic['uname'])
                        comp_weight[f'comp{i_sig + 1}'].append(int(sig_dic['weight']))
            # TODO add the display of the complete chain measurement (see comment below)
            #df = pd.DataFrame({'id': channel_ids, 'comp1 type': 0, 'comp1 name': 0, 'comp1 weight': 0, 'comp2 type': 0, 'comp2 name': 0, 'comp2 weight': 0, 'comp3 type': 0, 'comp3 name': 0, 'comp3 weight': 0, 'type complete chain': 0, 'name': 0})
            df = pd.DataFrame({'id': channel_ids, 'comp1 type': comp_type['comp1'], 'comp1 name': comp_name['comp1'], 'comp1 weight': comp_weight['comp1'], 'comp2 type': comp_type['comp2'], 'comp2 name': comp_name['comp2'], 'comp2 weight': comp_weight['comp2'], 'comp3 type': comp_type['comp3'], 'comp3 name': comp_name['comp3'], 'comp3 weight': comp_weight['comp3']})
        else:
            df = pd.DataFrame({'A': []})
        return df

    channel_data = get_channel_table()

    def highlight_cols(s):
        color = 'yellow'
        return 'background-color: %s' % color
    cont.table(channel_data.style.applymap(highlight_cols, subset=pd.IndexSlice[:, ['id']]))

    cont.subheader('Signal chain')
    signal_data = get_signal_chain_table()

    cont.table(signal_data.style.applymap(highlight_cols, subset=pd.IndexSlice[:, ['id']]))


# main page setup
page_configuration()

main_cont = st.container()
build_main_page(main_cont)


