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
from rnog_data.runtable import RUN_TYPES
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

page_name = 'runTable'

def color_survived(val):
    if val==0:
        color = "red"
        return f'color: {color}'
    else:
        return ''

def color_livetime(val):
    if val > 7100:
        color = "green"
        return f'color: {color}'
    else:
        return ''

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
    #main_cont.markdown(page_name)
    main_cont.subheader('Main pre-selection settings')
    selected_station = build_station_selection(main_cont)
    possible_flags = [rt.value for rt in RUN_TYPES]
    selected_flags = main_cont.multiselect('Select one/multiple run types:',possible_flags, default=possible_flags, key='flags')
    main_cont.markdown('**Select a time range:**')
    col_start1, col_start2, col_end1, col_end2 = main_cont.columns([1,1,1,1])
    start_date = col_start1.date_input('Start time:', value=datetime.utcnow()-timedelta(days=14), min_value=datetime(1969,1,1), key='start_date')
    start_time = col_start2.time_input('Start time:', value=time(0, 0, 0), label_visibility='hidden', key='start_time')
    start_date = datetime.combine(start_date, start_time)
    end_date = col_end1.date_input('End time:', value=datetime.utcnow(), key='end_date')
    end_time = col_end2.time_input('End time:', value=time(0, 0, 0), label_visibility='hidden', key='end_time')
    end_date = datetime.combine(end_date, end_time)
    #main_cont.subheader('secondary settings')
    #col_1, col_2, col_3 = main_cont.columns([1,1,1])
    #possible_trigger = ['rf0', 'rf1', 'ext', 'pps', 'soft']
    #trigger = col_1.multiselect('Select which trigger are enabled:', possible_trigger, default=possible_trigger, key='trigger')
    #duration = col_2.number_input('Select the minimal run duration (min):', min_value=0.0, max_value=120.0, key='duration')
    #duration = duration * units.minute
    #firmware = build_firmware_select(col_3)


    newnames = {
        'trigger_rf0_enabled': "RF0",
        'trigger_rf1_enabled': "RF1",
        'trigger_ext_enabled': "PA",
        'trigger_pps_enabled': "PPS",
        'trigger_soft_enabled': "soft"}

    # load all fitting runs into the cache
    #@st.experimental_memo
    def load_run_data_into_cache():
        run_info = load_runs(selected_station, start_date, end_date, selected_flags)
        run_info = run_info.drop(labels=['_id', 'path', 'quality_flag', 'quality_comment'], axis=1) # TODO remove quality_flag and quality_comment globally from DB
        run_info = run_info.rename(columns=newnames)
        run_info.insert(0, 'run_type', run_info.pop('run_type'))
        return run_info

    # display them as a df

    run_df = load_run_data_into_cache()

    gb = GridOptionsBuilder.from_dataframe(run_df)
    gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True, hide=True)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    use_columns = ['run_type', 'station', 'run', 'time_start', 'duration', 'trigger_rate', 'daq_config_comment', 'checks_failed'] #'RF0', 'RF1', 'PA', 'PPS', 'soft', 'n_events_recorded'

    gb.configure_columns(use_columns, hide=False)
    #gb.configure_side_bar()
    gridoptions = gb.build()

    with main_cont:
        response = AgGrid(
            run_df,
            height=600,
            gridOptions=gridoptions,
            enable_enterprise_modules=False,#True,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            fit_columns_on_grid_load=False,
            header_checkbox_selection_filtered_only=True,
            use_checkbox=True)
        st.write(f"Total number of rows: {len(run_df)}")
        st.write("(If no table is displayed, toggle text size (using 'cmd' +  '+/-') ... sorry, still trying to fix this :)")

    v = response['selected_rows'] 
    if v:
        print(v)
        outdf = pd.DataFrame(v)
        #outdf.insert(2, 'checks_failed', outdf.pop('checks_failed'))
        outdf = outdf.drop(labels=["_selectedRowNodeInfo"], axis=1)
        styler = outdf.style.applymap(color_livetime, subset=pd.IndexSlice[:, ['duration']])
        styler.applymap(color_survived, subset=["RF0","RF1", "PA", "PPS", "soft"])
        main_cont.dataframe(styler, use_container_width=True)

        main_cont.download_button('DOWNLOAD TABLE AS CSV', outdf.to_csv(), file_name='runtable_data.csv', mime='text/csv')


# main page setup
page_configuration()

main_container = st.container()
build_main_page(main_container)
