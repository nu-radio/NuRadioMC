from NuRadioReco.detector.db_mongo_write import Database
from NuRadioReco.detector.webinterface import config

db = Database(database_connection=config.DATABASE_TARGET)


def select_cable(page_name, main_container, warning_container_top):
    col1_cable, col2_cable = main_container.columns([1, 1])

    # load cable names from the database
    cable_names_db = db.get_object_names(page_name)

    cable_names_db.insert(0, 'new cable')
    cable_name = col1_cable.selectbox('Select existing cable or enter unique name of new cable:', cable_names_db)
    disable_text_input = False
    if cable_name == 'new cable':
        disable_text_input = False
    new_cable_name = col2_cable.text_input('Select existing cable :', placeholder='new cable name', label_visibility='hidden', disabled=disable_text_input)
    if cable_name == 'new cable':
        selected_cable_name = new_cable_name
    else:
        selected_cable_name = cable_name
    main_container.markdown(selected_cable_name)
    print('cable_name', selected_cable_name)
    if selected_cable_name in cable_names_db:
        warning_container_top.warning(f'You are about to override the {page_name} with the name \'{selected_cable_name}\'!')

    return selected_cable_name


def validate_global_cable_old(container_bottom, cable_type, cable_sta, cable_cha, channel_working, Sdata_validated_magnitude, Sdata_validated_phase, uploaded_data_magnitude, uploaded_data_phase):
    disable_insert_button = True
    name_validation = False
    # if nothing is chosen, a warning is given and the INSERT button stays disabled
    if cable_type == 'Choose an option' or cable_sta == 'Choose an option' or cable_cha == 'Choose an option':
        container_bottom.error('Not all cable options are selected')
        name_validation = False
    else:
        name_validation = True

    if name_validation:
        if not Sdata_validated_magnitude and uploaded_data_magnitude is not None:
            container_bottom.error('There is a problem with the magnitude input data')
            disable_insert_button = True

        if not Sdata_validated_phase and uploaded_data_phase is not None:
            container_bottom.error('There is a problem with the phase input data')
            disable_insert_button = True

        if Sdata_validated_magnitude and Sdata_validated_phase:
            disable_insert_button = False
            container_bottom.success('All inputs validated')

        if not channel_working:
            container_bottom.warning('The channel is set to not working')
            disable_insert_button = False
            container_bottom.success('All inputs validated')
    else:
        disable_insert_button = True

    return disable_insert_button


def validate_global_cable(container_bottom, cable_name, channel_working, Sdata_validated_magnitude, Sdata_validated_phase, uploaded_data_magnitude, uploaded_data_phase):
    disable_insert_button = True
    name_validation = False
    # if nothing is chosen, a warning is given and the INSERT button stays disabled
    if cable_name == '':
        container_bottom.error('No cable name is given!')
        name_validation = False
    else:
        name_validation = True

    if name_validation:
        if not Sdata_validated_magnitude and uploaded_data_magnitude is not None:
            container_bottom.error('There is a problem with the magnitude input data')
            disable_insert_button = True

        if not Sdata_validated_phase and uploaded_data_phase is not None:
            container_bottom.error('There is a problem with the phase input data')
            disable_insert_button = True

        if Sdata_validated_magnitude and Sdata_validated_phase:
            disable_insert_button = False
            container_bottom.success('All inputs validated')

        if not channel_working:
            container_bottom.warning('The channel is set to not working')
            disable_insert_button = False
            container_bottom.success('All inputs validated')
    else:
        disable_insert_button = True

    return disable_insert_button


def insert_cable_to_db(page_name, s_name, cable_name, data_m, data_p, input_units, working, primary, protocol):
    if not working:
        db.set_not_working(page_name, cable_name, primary)
    else:
        db.cable_add_Sparameter(page_name, cable_name, [s_name], data_m, data_p, input_units, primary, protocol)
