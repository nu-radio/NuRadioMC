import time
import streamlit as st
import pandas as pd
from NuRadioReco.detector.webinterface.utils.page_config import page_configuration
from NuRadioReco.detector.webinterface.utils.helper_runTable import get_firmware_from_db, load_runs, get_station_ids_from_db
from NuRadioReco.utilities import units
from datetime import datetime, timedelta
from datetime import time

page_name = 'runTable'


def delete_cache():
    st.experimental_memo.clear()
    #st.experimental_rerun()


def build_station_selection(cont):
    station_ids = get_station_ids_from_db()
    station_list = []
    for sta_id in station_ids:
        station_list.append(f'station {sta_id}')
    selected_stations = cont.multiselect('Select one/multiple stations:', station_list, default=station_list, key='station')  # ,on_change=delete_cache())

    selected_station_ids = []
    for sel_sta in selected_stations:
        selected_station_ids.append(int(sel_sta[len('station '):]))

    return selected_station_ids


def build_firmware_select(cont):
    firmware = get_firmware_from_db()
    selected_firmware = cont.multiselect('Select one/multiple firmware versions:', firmware, default=firmware, key='firmware')

    return selected_firmware


def build_main_page(main_cont):
    main_cont.title('Summary of data runs transferred from Greenland:')
    main_cont.markdown(page_name)
    main_cont.subheader('main settings')
    selected_station = build_station_selection(main_cont)
    main_cont.markdown('**Select a time range:**')
    col_start1, col_start2, col_end1, col_end2 = main_cont.columns([1,1,1,1])
    start_date = col_start1.date_input('Start time:', value=datetime.utcnow()-timedelta(days=14), key='start_date')
    start_time = col_start2.time_input('Start time:', value=time(0, 0, 0), label_visibility='hidden', key='start_time')
    start_date = datetime.combine(start_date, start_time)
    end_date = col_end1.date_input('End time:', value=datetime.utcnow(), key='end_date')
    end_time = col_end2.time_input('End time:', value=time(0, 0, 0), label_visibility='hidden', key='end_time')
    end_date = datetime.combine(end_date, end_time)
    possible_flags = ['calibration', 'physics', 'high_rate', 'daq_issue']
    selected_flags = main_cont.multiselect('Select one/multiple quality flags:',possible_flags, default=possible_flags, key='flags')
    main_cont.subheader('secondary settings')
    col_1, col_2, col_3 = main_cont.columns([1,1,1])
    possible_trigger = ['rf0', 'rf1', 'ext', 'pps', 'soft']
    trigger = col_1.multiselect('Select which trigger are enabled:', possible_trigger, default=possible_trigger, key='trigger')
    duration = col_2.number_input('Select the minimal run duration (min):', min_value=0.0, max_value=120.0, key='duration')
    duration = duration * units.minute
    firmware = build_firmware_select(col_3)

    # load all fitting runs into the cache
    #@st.experimental_memo
    def load_run_data_into_cache():
        run_info = load_runs(selected_station, start_date, end_date, selected_flags, trigger, duration, firmware)
        run_info = run_info.drop(labels=['_id', 'path'], axis=1)
        run_info = run_info.rename(columns={'trigger_rf0_enabled': 'rf0', 'trigger_rf1_enabled': 'rf1', 'trigger_ext_enabled': 'ext', 'trigger_pps_enabled': 'pps', 'trigger_soft_enabled': 'soft'})
  
        return run_info

    # display them as a df

    run_df = load_run_data_into_cache()

    def highlight_cols(s):
        color = 'yellow'
        return 'background-color: %s' % color

    main_cont.dataframe(run_df.style.applymap(highlight_cols, subset=pd.IndexSlice[:, ['station']]), use_container_width=True)

    main_cont.download_button('DOWNLOAD TABLE AS CSV', run_df.to_csv(), file_name='runtable_data.csv', mime='text/csv')


# main page setup
page_configuration()

main_container = st.container()
build_main_page(main_container)
