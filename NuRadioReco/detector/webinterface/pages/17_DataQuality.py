import copy
import time
import streamlit as st
import pandas as pd
from plotly import subplots
import plotly.graph_objs as go
from NuRadioReco.detector.webinterface.utils.page_config import page_configuration
#from NuRadioReco.detector.webinterface.utils.helper_runTable import get_firmware_from_db, load_runs, get_station_ids_from_db
from NuRadioReco.utilities import units
from datetime import datetime, timedelta
from datetime import time
import json
from rnog_data.runtable import RunTable

page_name = 'Data Quality'
new_dq_label = "<Insert new data quality analysis>"

def delete_cache():
    st.experimental_memo.clear()
    #st.experimental_rerun()


def build_main_page(main_cont):
    runtab = RunTable()

    main_cont.title('Management area for analysis results on data quality')
    main_cont.subheader('Choose existing data quality analysis or enter a new one')

    # Create selectbox
    options = [new_dq_label] + runtab.find_quality_checks()
    column_select_analysis, column_new_analysis = main_cont.columns([1, 1])
    selection = column_select_analysis.selectbox("Select quality analysis", options=options)

    # Create text input for user entry
    disable_text_input = True
    if selection == new_dq_label:
        disable_text_input = False
    otherOption = column_new_analysis.text_input("Enter name for new quality analysis", disabled=disable_text_input)

    # Just to show the selected option
    if selection != new_dq_label:
        quality_name = selection
        #st.info(f":white_check_mark: The selected option is {selection} ")
    else:
        quality_name = otherOption
        #st.info(f":white_check_mark: The written option is {otherOption} ")

    if quality_name in runtab.find_quality_checks():
        result =  runtab.get_quality_check_data(quality_name)
        name = result["quality_check_name"]
        link = result["link"]
        description = result["description"]
    else:
        name = ""
        link = ""
        description = ""

    quality_link = main_cont.text_input("Link to documentation of quality check", value=link, placeholder="provide link to wiki (github?)")
    quality_description = main_cont.text_area("Short description of quality check", value=description, placeholder="provide a short description")
    
    if quality_name and quality_link and quality_description:
        if quality_name not in runtab.find_quality_checks():
            register_button = main_cont.button(label="register")
            if register_button:
                runtab.register_quality_check(name=quality_name, description=quality_description, link=quality_link)
                main_cont.info("... registered!")    

    n_entries = 0
    # display page
    main_cont.subheader('Display and download data quality analysis results')
    if quality_name:
        result = runtab.get_quality_check_data(quality_name)
        if result is not None:

            # display found table
            for key in result:
                if key in ["data", "_id"]:
                    continue
                else:
                    main_cont.write(f"{result[key]}")
            main_cont.dataframe(result["data"])
            n_entries = len(result['data'])

            # download found table
            out_csv = []
            for key in result:
                if key in ["data", "_id"]:
                    continue
                else:
                    fstr = result[key].replace('\n','\n### ')
                    out_csv.append(f"### {fstr}")
            out_csv.append(pd.DataFrame(result["data"]).to_csv(index=False))

            main_cont.download_button(
                label="Download .csv",
                file_name=f"quality_check__{quality_name.replace(' ', '_')}.csv",
                mime="application/csv",
                data="\n".join(out_csv),
            )



    # new data upload
    main_cont.subheader('Upload additional data quality analysis results')
    disable_upload = False
    if n_entries > 0:
        disable_upload =True
    uploaded_file = main_cont.file_uploader("Upload file", type=['csv'], accept_multiple_files=False,key="quality_check_file_ploader")
    quality_data_enter_button = main_cont.button(label="enter data to database", disabled=disable_upload)
    if quality_name and quality_data_enter_button and uploaded_file is not None:
        main_cont.info("Uploading quality analysis to database")
        runtab.insert_quality_check_data(uploaded_file, quality_data_select)


# main page setup
page_configuration()

main_container = st.container()
build_main_page(main_container)
