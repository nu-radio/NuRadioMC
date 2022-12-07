import copy
import time
import streamlit as st
import pandas as pd
from plotly import subplots
import plotly.graph_objs as go
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
        if run_info != []:
            db_station = []
            db_run = []
            db_start = []
            db_end = []
            db_duration = []
            db_transfer_sample = []
            db_trigger_rate = []
            db_n_events_g = []
            db_n_events_t = []
            db_rf0_enabled = []
            db_rf1_enabled = []
            db_ext_enabled = []
            db_pps_enabled = []
            db_soft_enabled = []
            db_soft_rate = []
            db_flag = []
            db_firmware = []
            db_daq_config_comment = []
            for entry in run_info:
                db_station.append(entry['station'])
                db_run.append(entry['run'])
                db_start.append(entry['time_start'])
                db_end.append(entry['time_end'])
                db_duration.append(entry['duration'])
                db_transfer_sample.append(entry['transfer_subsampling'])
                db_trigger_rate.append(entry['trigger_rate'])
                db_n_events_g.append(entry['n_events_recorded'])
                db_n_events_t.append(entry['n_events_transferred'])
                db_rf0_enabled.append(entry['trigger_rf0_enabled'])
                db_rf1_enabled.append(entry['trigger_rf1_enabled'])
                db_ext_enabled.append(entry['trigger_ext_enabled'])
                db_pps_enabled.append(entry['trigger_pps_enabled'])
                db_soft_enabled.append(entry['trigger_soft_enabled'])
                db_soft_rate.append(entry['soft_trigger_rate'])
                db_flag.append(entry['quality_flag'])
                db_firmware.append(entry['firmware_version'])
                db_daq_config_comment.append(entry['daq_config_comment'])

            df = pd.DataFrame({'station': db_station, 'run': db_run, 'start': db_start, 'end': db_end, 'duration (min)': db_duration, 'transfer subsampling': db_transfer_sample, 'trigger rate': db_trigger_rate, 'n events (greenland)': db_n_events_g, 'n events (transferred)': db_n_events_t, 'rf0': db_rf0_enabled, 'rf1': db_rf1_enabled, 'ext': db_ext_enabled, 'pps': db_pps_enabled, 'soft': db_soft_enabled, 'soft trigger rate': db_soft_rate, 'quality flag': db_flag, 'firmware': db_firmware, 'daq config comment': db_daq_config_comment})

        else:
            df = pd.DataFrame({'station': [None], 'run': [None], 'start': [None], 'end': [None], 'duration (min)': [None], 'transfer subsampling': [None], 'trigger rate': [None], 'n events (greenland)': [None], 'n events (transferred)': [None], 'rf0': [None], 'rf1': [None], 'ext': [None], 'pps': [None], 'soft': [None], 'soft trigger rate': [None], 'quality flag': [None], 'firmware': [None], 'daq config comment': [None]})

        return df

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
