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
    cont.subheader('channel info')
    # load and cache the data (is only loaded once)
    @st.experimental_memo
    def get_channel_table():
        if 'channels' in list(station_info[selected_station_id].keys()):
            # transform the channel data in a dataframe
            channel_dic = station_info[selected_station_id]['channels']

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
            df = pd.DataFrame({'id': channel_ids, 'antenna name': antenna_name, 'antenna position (x,y,z)': antenna_position, 'orientation (theta, phi)': orientation, 'rotation (theta, phi)': rotation, 'antenna type': antenna_type, 'commission': commission_time, 'decommission': decommission_time, 'comment': comments})
            print(df)

        else:
            df = pd.DataFrame({'A': []})

        return df

    channel_data = get_channel_table()

    def highlight_cols(s):
        color = 'yellow'
        return 'background-color: %s' % color
    cont.table(channel_data.style.applymap(highlight_cols, subset=pd.IndexSlice[:, ['id']]))

    cont.subheader('Signal chain')
    # display the information
    # current favourite:
    # station infos as disabled text inputs, or just markdowns?
    # channel infos as pd dataframe -> can search the data frame, can resize the column, can highlight stuff
    # two ideas:
    # 0) or use st.dataframe (can use the highlight tool from pandas)
    # 1) put everything in a table and display this table
    # 2) second use the inputs and disable them to display the data
    # IMPORTANT! I don't want to change something on this page


# main page setup
page_configuration()

main_cont = st.container()
build_main_page(main_cont)


