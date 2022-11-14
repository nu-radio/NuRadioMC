import streamlit as st
from NuRadioReco.detector.webinterface.utils.page_config import page_configuration
from NuRadioReco.detector.webinterface.utils.helper_display_station import build_station_selection, load_station_position_info, get_all_measurement_names
import pandas as pd
from datetime import datetime
from datetime import time

def build_main_page(cont):
    cont.title('Display station information')
    cont.subheader('select station')
    collection_name = 'station_rnog'
    # select the station which should be displayed
    selected_station_name, selected_station_id = build_station_selection(cont, collection_name)

    # makes sure the data is reloaded if collection or station is changed, otherwise the date stays in the cache
    if 'station_id' not in st.session_state:
        st.session_state['station_id'] = 0

    if st.session_state.station_id != selected_station_id:
        st.session_state.station_id = selected_station_id
        st.experimental_memo.clear()
        st.experimental_rerun()

    # get the station position

    cont.subheader('station position info')
    coll_pos_select1, coll_pos_select2 = cont.columns([1, 1])
    selected_primary_time = coll_pos_select1.date_input('Primary time', value=datetime.utcnow())
    selected_primary_time = datetime.combine(selected_primary_time, time(12, 0, 0))
    measurement_names_db = list(get_all_measurement_names())
    measurement_names_db.insert(0, 'not specified')
    selected_measurement_name = coll_pos_select2.selectbox('Measurement name:', options=measurement_names_db)

    if 'measurement_time_key' not in st.session_state:
        st.session_state['measurement_time_key'] = ''

    if 'primary_time_key' not in st.session_state:
        st.session_state['primary_time_key'] = None

    if st.session_state.measurement_time_key != selected_measurement_name:
        st.session_state.measurement_time_key = selected_measurement_name
        st.experimental_memo.clear()
        st.experimental_rerun()

    if st.session_state.primary_time_key != selected_primary_time:
        st.session_state.primary_time_key = selected_primary_time
        st.experimental_memo.clear()
        st.experimental_rerun()

    # load the position information into the cache
    @st.experimental_memo
    def get_pos_info():
        position_info = load_station_position_info(station_id=selected_station_id, primary_time=selected_primary_time, measurement_name=selected_measurement_name)
        meas_name = None
        pri_times = [{'start': None, 'end': None}]
        pos = [None, None, None]
        if position_info != []:
            meas_name = position_info[0]["measurements"]["measurement_name"]
            pos = position_info[0]["measurements"]["position"]
            pri_times = position_info[0]["measurements"]["primary_measurement"]
        return {'name': meas_name, 'position': pos, 'primary': pri_times}

    pos_info = get_pos_info()

    cont.markdown('**measurement name**')
    cont.text(f'{pos_info["name"]}')
    cont.markdown('**primary times**')
    for times in pos_info['primary']:
        coll1_pri, coll2_pri = cont.columns([1,1])
        if times["start"] is None:
            coll1_pri.code(f'start: None')
            coll2_pri.code(f'end: None')
        else:
            coll1_pri.code(f'start: {times["start"].date()} {times["start"].time().strftime("%H:%M:%S")}')
            coll2_pri.code(f'end: {times["end"].date()} {times["end"].time().strftime("%H:%M:%S")}')

    cont.markdown('**position**')
    coll1, coll2, coll3 = cont.columns([1, 1, 1])
    coll1.code(f'x: {pos_info["position"][0]}')
    coll2.code(f'y: {pos_info["position"][1]}')
    coll3.code(f'z: {pos_info["position"][2]}')

    # cont.markdown('**comments:**')
    # if 'station_comment' in station_info[selected_station_id].keys():
    #     comment = f'{station_info[selected_station_id]["station_comment"]}'
    #     if comment == '':
    #         comment = 'No comments'
    # else:
    #     comment = 'No comments'
    # cont.text(comment)
    #
    # cont.subheader('channel info')
    #
    # channel_dic = station_info[selected_station_id]['channels']
    #
    # # load and cache the channel information (is only loaded once)
    # @st.experimental_memo
    # def get_channel_table():
    #     if 'channels' in list(station_info[selected_station_id].keys()):
    #         # transform the channel data in a dataframe
    #         antenna_name = []
    #         antenna_position = []
    #         orientation = []
    #         rotation = []
    #         antenna_type = []
    #         commission_time = []
    #         decommission_time = []
    #         comments = []
    #         channel_ids = []
    #         for cha in channel_dic.keys():
    #             channel_ids.append(cha)
    #             antenna_name.append(channel_dic[cha]['ant_name'])
    #             antenna_position.append([float(channel_dic[cha]['ant_position'][0]), float(channel_dic[cha]['ant_position'][1]), float(channel_dic[cha]['ant_position'][2])])
    #             orientation.append([float(channel_dic[cha]['ant_ori_theta']), float(channel_dic[cha]['ant_ori_phi'])])
    #             rotation.append([float(channel_dic[cha]['ant_rot_theta']), float(channel_dic[cha]['ant_rot_phi'])])
    #             antenna_type.append(channel_dic[cha]['type'])
    #             commission_time.append(channel_dic[cha]['commission_time'].date())
    #             decommission_time.append(channel_dic[cha]['decommission_time'].date())
    #             if 'channel_comment' in channel_dic[cha].keys():
    #                 comments.append(channel_dic[cha]['channel_comment'])
    #             else:
    #                 comments.append('')
    #         df = pd.DataFrame({'id': channel_ids, 'antenna name': antenna_name, 'antenna type': antenna_type, 'antenna position (x,y,z)': antenna_position, 'orientation (theta, phi)': orientation, 'rotation (theta, phi)': rotation, 'commission': commission_time, 'decommission': decommission_time, 'comment': comments})
    #     else:
    #         df = pd.DataFrame({'A': []})
    #
    #     return df
    #
    # # load and cache the signal chain information (is only loaded once)
    # #TODO add the complete chain measurement
    # @st.experimental_memo
    # def get_signal_chain_table():
    #     if 'channels' in list(station_info[selected_station_id].keys()):
    #         # transform the channel data in a dataframe
    #         channel_ids = []
    #         comp_type = {'comp1': [], 'comp2': [], 'comp3': []}
    #         comp_name = {'comp1': [], 'comp2': [], 'comp3': []}
    #         comp_weight = {'comp1': [], 'comp2': [], 'comp3': []}
    #         for cha in channel_dic.keys():
    #             channel_ids.append(cha)
    #             # transform the signal chain data
    #             signal_chain = channel_dic[cha]['signal_ch']
    #             # TODO: also consider the case where only the complete chain is in the db
    #             if len(signal_chain) < 3:
    #                 for i_sig, sig_dic in enumerate(signal_chain):
    #                     comp_type[f'comp{i_sig + 1}'].append(sig_dic['type'])
    #                     comp_name[f'comp{i_sig + 1}'].append(sig_dic['uname'])
    #                     comp_weight[f'comp{i_sig + 1}'].append(int(sig_dic['weight']))
    #                 comp_type[f'comp{3}'].append(None)
    #                 comp_name[f'comp{3}'].append(None)
    #                 comp_weight[f'comp{3}'].append(None)
    #             else:
    #                 for i_sig, sig_dic in enumerate(signal_chain):
    #                     comp_type[f'comp{i_sig + 1}'].append(sig_dic['type'])
    #                     comp_name[f'comp{i_sig + 1}'].append(sig_dic['uname'])
    #                     comp_weight[f'comp{i_sig + 1}'].append(int(sig_dic['weight']))
    #         # TODO add the display of the complete chain measurement (see comment below)
    #         #df = pd.DataFrame({'id': channel_ids, 'comp1 type': 0, 'comp1 name': 0, 'comp1 weight': 0, 'comp2 type': 0, 'comp2 name': 0, 'comp2 weight': 0, 'comp3 type': 0, 'comp3 name': 0, 'comp3 weight': 0, 'type complete chain': 0, 'name': 0})
    #         df = pd.DataFrame({'id': channel_ids, 'comp1 type': comp_type['comp1'], 'comp1 name': comp_name['comp1'], 'comp1 weight': comp_weight['comp1'], 'comp2 type': comp_type['comp2'], 'comp2 name': comp_name['comp2'], 'comp2 weight': comp_weight['comp2'], 'comp3 type': comp_type['comp3'], 'comp3 name': comp_name['comp3'], 'comp3 weight': comp_weight['comp3']})
    #     else:
    #         df = pd.DataFrame({'A': []})
    #     return df
    #
    # channel_data = get_channel_table()
    #
    # def highlight_cols(s):
    #     color = 'yellow'
    #     return 'background-color: %s' % color
    # # cont.table(channel_data.style.applymap(highlight_cols, subset=pd.IndexSlice[:, ['id']]))
    # cont.dataframe(channel_data.style.applymap(highlight_cols, subset=pd.IndexSlice[:, ['id']]), use_container_width=True)
    #
    # cont.subheader('signal chain')
    # signal_data = get_signal_chain_table()
    #
    # cont.dataframe(signal_data.style.applymap(highlight_cols, subset=pd.IndexSlice[:, ['id']]), use_container_width=True)
    # #, height=int(len(signal_data.index+1)/3*200)

# main page setup
page_configuration()

main_cont = st.container()
build_main_page(main_cont)


