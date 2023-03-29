import time
import streamlit as st
from NuRadioReco.detector.webinterface.utils.page_config import page_configuration
from NuRadioReco.detector.webinterface.utils.helper import build_success_page
from NuRadioReco.detector.webinterface.utils.helper_station import load_station_ids, load_measurement_names, insert_device_position_to_db, load_collection_information
from datetime import datetime
from datetime import time

page_name = 'station'
collection_name = 'station_rnog'


def validate_inputs(container_bottom, station_id, measurement_name, device_id):
    device_id_correct = False
    station_in_db = False
    measurement_name_correct = False
    device_in_db = False

    disable_insert_button = True

    # validate that the device id is correct
    if device_id >= 0:
        device_id_correct = True
    else:
        container_bottom.info('For this measurement, all devices are already int the database.')

    # validate that a valid measurement name is given
    if measurement_name != '' and measurement_name != 'new measurement':
        measurement_name_correct = True
    else:
        container_bottom.error('Measurement name is not valid.')

    # check if the station is available in the database
    if station_id in load_station_ids():
        station_in_db = True
    else:
        container_bottom.error('There is no corresponding entry for the station in the database. Please insert the general station information first!')

    # check that the device is not in the database for this measurement
    if device_id in measurement_device_ids_db:
        container_bottom.error('There is already a device position information saved in the database for this measurement name and this device id.')
    else:
        device_in_db = True

    if device_id_correct and station_in_db and measurement_name_correct and device_in_db:
        disable_insert_button = False

    return disable_insert_button


def build_device_position_page(main_cont):
    cont_device = main_cont.container()

    device_id = cont_device.selectbox('Select a device id:', device_help)
    if device_id is None:
        selected_device_id = -10
    else:
        selected_device_id = int(device_id)

    # primary measurement?
    primary = main_cont.checkbox('Is this the primary measurement?', value=True)

    # measurement time
    measurement_time = main_cont.date_input('Enter the time of the measurement:', value=datetime.utcnow(), help='The date when the measurement was conducted.')
    measurement_time = datetime.combine(measurement_time, time(0, 0, 0))

    position = None
    rot = None
    ori = None
    if selected_device_id >= 0:
        if device_dic[selected_device_id] in ['Solar panel 1', 'Solar panel 2', 'wind-turbine', 'daq box']:
            default_dev_pos1 = [0, 0, 0]
            default_dev_pos2 = [0, 0, 0]

            main_cont.markdown('First measurement point:')
            col1_pos, col2_pos, col3_pos = main_cont.columns([1, 1, 1])
            x_pos_dev1 = col1_pos.text_input('x position', key='x_dev1', value=default_dev_pos1[0])
            y_pos_dev1 = col2_pos.text_input('y position', key='y_dev1', value=default_dev_pos1[1])
            z_pos_dev1 = col3_pos.text_input('z position', key='z_dev1', value=default_dev_pos1[2])
            position1 = [float(x_pos_dev1), float(y_pos_dev1), float(z_pos_dev1)]

            main_cont.markdown('Second measurement point:')
            col11_pos, col21_pos, col31_pos = main_cont.columns([1, 1, 1])
            x_pos_dev2 = col11_pos.text_input('x position', key='x_dev2', value=default_dev_pos2[0])
            y_pos_dev2 = col21_pos.text_input('y position', key='y_dev2', value=default_dev_pos2[1])
            z_pos_dev2 = col31_pos.text_input('z position', key='z_dev2', value=default_dev_pos2[2])
            position2 = [float(x_pos_dev2), float(y_pos_dev2), float(z_pos_dev2)]
            position = [position1, position2]
        else:
            default_dev_pos = [0, 0, 0]
            default_rot = {'theta': 0, 'phi': 0}
            default_ori = {'theta': 0, 'phi': 0}

            col1_pos, col2_pos, col3_pos = main_cont.columns([1, 1, 1])
            x_pos_ant = col1_pos.text_input('x position', key='x_antenna', value=default_dev_pos[0])
            y_pos_ant = col2_pos.text_input('y position', key='y_antenna', value=default_dev_pos[1])
            z_pos_ant = col3_pos.text_input('z position', key='z_antenna', value=default_dev_pos[2])
            position = [float(x_pos_ant), float(y_pos_ant), float(z_pos_ant)]

            # input the orientation, rotation of the antenna; if the channel already exist, insert the values from the database
            col1a, col2a, col3a, col4a = main_cont.columns([1, 1, 1, 1])
            ant_ori_theta = col1a.text_input('orientation (theta):', value=default_ori['theta'])
            ant_ori_phi = col2a.text_input('orientation (phi):', value=default_ori['phi'])
            ant_rot_theta = col3a.text_input('rotation (theta):', value=default_rot['theta'])
            ant_rot_phi = col4a.text_input('rotation (phi):', value=default_rot['phi'])
            ori = {'theta': float(ant_ori_theta), 'phi': float(ant_ori_phi)}
            rot = {'theta': float(ant_rot_theta), 'phi': float(ant_rot_phi)}

    # container for warnings/infos at the botton
    cont_warning_bottom = main_cont.container()

    disable_insert_button = validate_inputs(cont_warning_bottom, selected_station_id, measurement_name, selected_device_id)

    insert_device = main_cont.button('INSERT DEVICE TO DB', disabled=disable_insert_button)

    cont_device_warning = main_cont.container()

    if insert_device:
        insert_device_position_to_db(selected_station_id, selected_device_id, measurement_name, measurement_time, position, ori, rot, primary)
        main_cont.empty()
        st.session_state.key = '1'
        st.experimental_rerun()


# main page setup
page_configuration()

# initialize the session key (will be used to display different pages depending on the button clicked)
if 'key' not in st.session_state:
    st.session_state['key'] = '0'

main_container = st.container()

main_container.title('Add device position and orientation information')
main_container.markdown(page_name)

if 'insert_device' not in st.session_state:
    st.session_state.insert_device = False

# select a unique name for the measurement (survey_01, tape_measurement, ...)
col1_name, col2_name = main_container.columns([1, 1])
measurement_list = load_measurement_names('device_position')
measurement_list.insert(0, 'new measurement')
selected_name = col1_name.selectbox('Select or enter a unique name for the measurement:', measurement_list)
disabled_name_input = True
if selected_name == 'new measurement':
    disabled_name_input = False
name_input = col2_name.text_input('Select or enter a unique name for the measurement:', disabled=disabled_name_input, label_visibility='hidden')
measurement_name = selected_name
if measurement_name == 'new measurement':
    measurement_name = name_input

# enter the information for the station
cont_warning_top = main_container.container()
station_list = ['Station 11 (Nanoq)', 'Station 12 (Terianniaq)', 'Station 13 (Ukaleq)', 'Station 14 (Tuttu)', 'Station 15 (Umimmak)', 'Station 21 (Amaroq)', 'Station 22 (Avinngaq)', 'Station 23 (Ukaliatsiaq)',
                'Station 24 (Qappik)', 'Station 25 (Aataaq)']
selected_station = main_container.selectbox('Select a station', station_list)
# get the name and id out of the string
selected_station_id = int(selected_station[len('Station '):len('Station ') + 2])

cont_warning_top = main_container.container()

main_container.subheader('Input device information')

# load the selected measurement information  (to display how many devices are inserted for this specific measurement)
device_info_db = load_collection_information('device_position', selected_station_id, measurement_name=measurement_name)

# extract the device ids from the measurement
measurement_device_ids_db = []
device_name_display = ''
device_dic = {1: 'Helper string B pulser', 0: 'Helper string C pulser', 2: 'Surface pulser', 101: 'Solar panel 1', 102: 'Solar panel 2', 103: 'wind-turbine', 100: 'daq box'}
for entry in device_info_db:
    measurement_device_ids_db.append(entry['measurements']['device_id'])
    device_name_display += f'{device_dic[entry["measurements"]["device_id"]]}, '
device_name_display = device_name_display[:-2]

main_container.info(f'Devices included in the database: {device_name_display}')

# create a list with device ids not in the database
device_help = []
for dev in device_dic.keys():
    if dev not in measurement_device_ids_db:
        device_help.append(dev)

device_container = st.container()  # container for the main part of the page (with all the input filed)
success_container = st.container()  # container to display the page when the data was submitted

if st.session_state.key == '0':
    build_device_position_page(device_container)  # after clicking the submit button, the session key is set to '1' and the page is rerun

if st.session_state.key == '1':
    build_success_page(success_container, 'device position')  # after clicking the 'add another measurement' button, the session key is set to '0' and the page is rerun
