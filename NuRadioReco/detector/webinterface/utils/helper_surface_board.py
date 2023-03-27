from NuRadioReco.detector.db_mongo_write import Database
from NuRadioReco.detector.webinterface import config

db = Database(database_connection=config.DATABASE_TARGET)


def select_surface(page_name, main_container, warning_container):
    col1_I, col2_I, col3_I, col4_I = main_container.columns([1, 1, 1, 1])

    selected_surface_name = ''
    surface_names = db.get_object_names(page_name)
    surface_names.insert(0, f'new {page_name}')

    surface_dropdown = col1_I.selectbox('Select existing board or enter unique name of new board:', surface_names)
    if surface_dropdown == f'new {page_name}':
        disable_new_input = False

        selected_surface_infos = []
    else:
        disable_new_input = True
        selected_surface_name = surface_dropdown
        warning_container.warning(f'You are about to override the {page_name} unit {surface_dropdown}!')

        # load all the information for this board
        selected_surface_infos = db.load_board_information(page_name, selected_surface_name, ['channel_id', 'measurement_temp'])

    new_board_name = col2_I.text_input('', placeholder=f'new unique board name', disabled=disable_new_input)
    if surface_dropdown == f'new {page_name}':
        selected_surface_name = new_board_name

    channel_numbers = ['Choose a channel-id', '0', '1', '2', '3', '4']
    # if an exiting drab is selected, change the default option to the saved IGLU
    if selected_surface_infos != []:
        cha_index = channel_numbers.index(str(selected_surface_infos[0]))
        channel_numbers.pop(cha_index)
        channel_numbers.insert(0, str(selected_surface_infos[0]))
    selected_channel_id = col3_I.selectbox('', channel_numbers)

    temp_list = ['room temp (20°C)', '-50°C', '-40°C', '-30°C', '-20°C', '-10°C', '0°C', '10°C', '30°C', '40°C']
    # if an exiting DRAB is selected, change the default option to the saved temperature
    if selected_surface_infos != []:
        if selected_surface_infos[1] == 20:
            saved_temp = 'room temp (20°C)'
        else:
            saved_temp = str(selected_surface_infos[1]) + '°C'
        temp_index = temp_list.index(saved_temp)
        temp_list.pop(temp_index)
        temp_list.insert(0, saved_temp)
    selected_Temp = col4_I.selectbox('', temp_list)
    if 'room temp' in selected_Temp:
        selected_Temp = int(selected_Temp[len('room temp ('):-3])
    else:
        selected_Temp = int(selected_Temp[:-2])

    return selected_surface_name, surface_dropdown, selected_channel_id, selected_Temp


def validate_global_surface(page_name, container_bottom, surface_name, new_surface_name, channel_id, channel_working, Sdata_validated, uploaded_data):
    disable_insert_button = True
    name_validation = False
    input_validation = False
    # if nothing is chosen, a warning is given and the INSERT button stays disabled
    if surface_name == '':
        container_bottom.error(f'{page_name} name is not set')
    elif surface_name == f'new {page_name}' and (new_surface_name is None or new_surface_name == ''):
        container_bottom.error(f'{page_name} name dropdown is set to \'new {page_name}\', but no new {page_name} name was entered.')
    else:
        name_validation = True

    if 'Choose' in channel_id and channel_working:
        container_bottom.error('Not all input options are entered.')
    else:
        input_validation = True

    if name_validation and input_validation:
        if not Sdata_validated and uploaded_data is not None:
            container_bottom.error('There is a problem with the input data')
            disable_insert_button = True
        elif Sdata_validated:
            disable_insert_button = False
            container_bottom.success('Input fields are validated')

        if not channel_working:
            container_bottom.warning('The channel is set to not working')
            disable_insert_button = False
            container_bottom.success('Input fields are validated')
    else:
        disable_insert_button = True

    return disable_insert_button


def insert_surface_to_db(page_name, s_names, surface_name, data, input_units, working, primary, protocol, channel_id, temp, measurement_time, time_delay):
    if not working:
        db.set_not_working(page_name, surface_name, primary, channel_id=int(channel_id))
    else:
        db.surface_add_Sparameters(page_name, s_names, surface_name, int(channel_id), temp, data, measurement_time, primary, time_delay, protocol, input_units)
