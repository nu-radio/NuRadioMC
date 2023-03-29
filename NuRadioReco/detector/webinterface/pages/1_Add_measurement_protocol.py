import streamlit as st
from NuRadioReco.detector.webinterface.utils.page_config import page_configuration
from NuRadioReco.detector.webinterface.utils.helper import build_success_page
from NuRadioReco.detector.webinterface.utils.helper_protocol import insert_measurement_protocol_into_db, check_inserted_config_information

page_name = 'measurement protocol'


def build_main_page(main_cont):
    main_cont.title('Add a measurement protocol')
    main_cont.markdown('On this page, you can insert a new measurement protocol.')
    main_cont.markdown(
        'The measurement protocols are used to keep track about how and by whom the measurement was conducted. This information will help if questions about or problems with these measurements arise.')
    link_protocols = 'https://radio.uchicago.edu/wiki/index.php/Measurement_protocols'
    main_cont.markdown(f'A complete list with all measurement protocols can be fund on the wiki: [measurement_protocols]({link_protocols})')
    main_cont.markdown('If you add a new measurement protocol to the database, please also add the protocol information to the wiki (link above).')

    protocol = main_cont.text_input('Please insert a new measurement protocol name.', help='Example name: Erlangen2020')

    # validate inputs
    validated = check_inserted_config_information(main_cont, protocol)

    # INSERT button
    upload_button = main_cont.button('INSERT TO DB', disabled=validated)

    # insert the data into the database and change to the success page by setting the session key
    if upload_button:
        insert_measurement_protocol_into_db(protocol)
        main_cont.empty()
        st.session_state.key = '1'
        st.experimental_rerun()


# main page setup
page_configuration()

# initialize the session key (will be used to display different pages depending on the button clicked)
if 'key' not in st.session_state:
    st.session_state['key'] = '0'

main_container = st.container()  # container for the main part of the page (with all the input filed)
success_container = st.container()  # container to display the page when the data was submitted

if st.session_state.key == '0':
    build_main_page(main_container)  # after clicking the submit button, the session key is set to '1' and the page is rerun

if st.session_state.key == '1':
    build_success_page(success_container, page_name)  # after clicking the 'add another measurement' button, the session key is set to '0' and the page is rerun
