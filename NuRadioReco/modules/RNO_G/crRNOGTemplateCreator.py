from __future__ import annotations
from typing import Any
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units
import astropy
import numpy as np
import NuRadioReco.detector.detector as detector
import NuRadioReco.modules.efieldToVoltageConverter
import logging
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
import datetime
from NuRadioReco.framework.event import Event
from NuRadioReco.framework.station import Station
from NuRadioReco.framework.channel import Channel
from NuRadioReco.framework.sim_station import SimStation
from NuRadioReco.framework.sim_channel import SimChannel
from NuRadioReco.framework.parameters import stationParameters
from NuRadioReco.framework.parameters import electricFieldParameters
from NuRadioReco.framework.electric_field import ElectricField
import pickle
import os
import NuRadioReco.modules.channelBandPassFilter
from NuRadioReco.modules.base.module import register_run


class crRNOGTemplateCreator:
    """
        Creates CR templates by assuming a gaussian function for the electric field

    """

    def __init__(self):
        self.logger = logging.getLogger("NuRadioReco.crRNOGTemplateCreator")
        self.__detector_file = None

        self.__template_run_id = None
        self.__template_channel_id = None
        self.__template_station_id = None

        self.__sampling_rate = None
        self.__template_sample_number = None

        self.__antenna_rotation = None
        self.__efield_width = None
        self.__efield_amplitudes = None
        self.__template_event_id = None

        self.__cr_zenith = None
        self.__cr_azimuth = None

        self.__debug = None

        self.__template_save_path = None

        self.__efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
        self.__hardwareResponseIncorporator = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()
        self.__channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()


    def begin(self, detector_file:str, template_save_path:str, debug:bool=False, logger_level:logging.Logger=logging.NOTSET) -> None:
        """
                begin method

                Parameters
                ----------
                detector_file: str
                    path to the detector file used for the template set
                template_save_path: str
                    path to a folder where the templates are stored
                debug: bool, default: False
                    enable/disable debug mode
                logger_level: str or int, optional
                    Set verbosity level for logger (default: logging.NOTSET)
                """

        self.__detector_file = detector_file

        self.logger.setLevel(logger_level)

        # set up the efield to voltage converter
        self.__efieldToVoltageConverter.begin(debug=debug)

        self.__debug = debug

        self.__template_save_path = template_save_path


    def set_template_parameter(self, template_run_id:list[int]=[0, 0, 0], template_event_id:list[int]=[0, 1, 2], template_station_id:list[int]=[101, 101, 101], 
                               template_channel_id:list[int]=[0, 0, 0], efield_width:list[float]=[5, 4, 2], antenna_rotation:list[float]=[160, 160, 160], 
                               efield_amplitudes:list[float]=[-0.2, 0.8], cr_zenith:list[float]=[55, 55, 55], cr_azimuth:list[float]=[0, 0, 0],
                               sampling_rate:float=3.2 * units.GHz, number_of_samples:int=2048) ->None:
        """
        set_parameter_templates method

        sets the parameter to create the template set

        Parameters
        ----------
        template_run_id: list of int, default: [0,0,0]
            run ids of the artificial templates
        template_event_id: list of int, default: [0,1,2]
            event ids of the artificial templates
        template_station_id: list of int, default: [101,101,101]
            station ids of the artificial templates
        template_channel_id: list of int, default: [0,0,0]
            channel ids of the artificial templates
        efield_width: list of int, default: [5,4,2]
            width (in samples) of the gaussian function used to create the Efield
        antenna_rotation: list of int, default: [160,160,160]
            rotation angle of the LPDA
        efield_amplitudes: list of float, default:[-0.2,0.8]
            array with the amplitudes of the Efield components [E_theta, E_phi]
        cr_zenith: list of int, default: [55,55,55]
            zenith angle of the cr for the template
        cr_azimuth: list of int, default: [0,0,0]
            azimuth angle of the cr for the template
        sampling_rate: float, default: 3.2
            sampling rate used to build the template
        number_of_samples: int, default: 2048
            number of samples used for the trace
        """

        self.__template_run_id = template_run_id
        self.__template_event_id = template_event_id
        self.__template_station_id = template_station_id
        self.__template_channel_id = template_channel_id

        self.__efield_width = efield_width
        self.__efield_amplitudes = efield_amplitudes
        self.__antenna_rotation = antenna_rotation

        self.__sampling_rate = sampling_rate
        self.__template_sample_number = number_of_samples

        self.__cr_zenith = cr_zenith
        self.__cr_azimuth = cr_azimuth


    @register_run()
    def run(self, template_filename:str='templates_cr_station_101.pickle', include_hardware_response:bool=True, hardware_response_source:str='json', 
            return_templates:bool=False, bandpass_filter:None|dict[str,Any]=None) -> None|list[Event]:
        """
        run method

        creates a pickle file with the Efield trace of the artificial templates

        Parameters
        ----------
        template_filename: str, default: 'templates_cr_station_101.pickle'
            filename of the pickle file that will be used to store the templates
        include_hardware_response: bool, default: True
            if true, the hardware response of the surface amps (hardwareResponseIncorporator) is applied
        hardware_response_source: str, default: "json"
            define if the hardware response is loaded from the json ('json') or from the database ('database')
        return_templates: bool, default: False
            if true, the template traces are returned in an addition to saving them in a pickle file
        bandpass_filter: dict, optional
            If a dictionary is given, a bandpass filter will be applied to the templates. The dictionary should hold all arguments that are needed for the channelBandPassFilter.
        
        Returns
        -------
        template_event: list of `NuRadioReco.framework.event.Event` or None
            If return templates is True, a list with the templates is returned.
        """

        # if no parameters are set, the standard parameters are used
        if self.__efield_width is None:
            self.logger.info("The default parameters are used for template creation.")
            self.set_template_parameter()

        template_events = []
        save_dic = {}
        for crz in list(set(self.__cr_zenith)):
            save_dic_help = {}
            for cra in list(set(self.__cr_azimuth)):
                # loop over the different antenna rotation angles:
                templates = {}
                for rid, eid, sid, cid, e_width, antrot, cr_zen, cr_az in zip(self.__template_run_id, self.__template_event_id, self.__template_station_id, self.__template_channel_id, self.__efield_width, self.__antenna_rotation, self.__cr_zenith, self.__cr_azimuth):
                    if cr_zen == crz and cr_az == cra:
                        # create the detector
                        det_temp = detector.generic_detector.GenericDetector(json_filename=self.__detector_file, antenna_by_depth=False, create_new=True, log_level='ERROR')
                        det_temp.update(datetime.datetime(2025, 10, 1))
                        det_temp.get_channel(sid, cid)['ant_rotation_phi'] = antrot

                        station_time = datetime.datetime(2025, 10, 1)

                        temp_evt = _create_Efield(det_temp, rid, eid, cid, sid, station_time, self.__template_sample_number, e_width,
                                                 self.__efield_amplitudes[1], self.__efield_amplitudes[0], cr_zen, cr_az, self.__sampling_rate, self.__debug)

                        self.__efieldToVoltageConverter.run(temp_evt, temp_evt.get_station(sid), det_temp)

                        if include_hardware_response:
                            if hardware_response_source == 'json':
                                self.logger.info("The placeholder hardware response from NuRadioMC is applied to the templates.")
                                self.__hardwareResponseIncorporator.run(temp_evt, temp_evt.get_station(sid), det_temp, sim_to_data=True)
                            elif hardware_response_source == 'database':
                                self.logger.info("The hardware response from the database is applied to the templates.")
                                # create a rno-g detetcor to load the hardware response
                                rnog_det = detector.Detector(source="rnog_mongo", log_level=logging.WARNING, always_query_entire_description=False,
                                                             database_connection='RNOG_public', select_stations=sid)
                                rnog_det.update(datetime.datetime(2023, 3, 4, 0, 0))
                                self.__hardwareResponseIncorporator.run(temp_evt, temp_evt.get_station(sid), rnog_det, sim_to_data=True)


                        if bandpass_filter is not None:
                            # apply the channelBandPassFilter
                            self.logger.info("The channelBandPassFilter is applied.")
                            self.__channelBandPassFilter.run(temp_evt, temp_evt.get_station(sid), det_temp, **bandpass_filter)

                        if self.__debug:
                            plt.plot(temp_evt.get_station(sid).get_channel(cid).get_times() / units.ns, temp_evt.get_station(sid).get_channel(cid).get_trace())
                            plt.xlabel('times [ns]')
                            plt.ylabel('amplitudes')
                            plt.show()
                            plt.plot(temp_evt.get_station(sid).get_channel(cid).get_frequencies() / units.MHz, np.abs(temp_evt.get_station(sid).get_channel(cid).get_frequency_spectrum()))
                            plt.xlabel('frequency [MHz]')
                            plt.ylabel('amplitudes')
                            plt.show()
                        template_events.append(temp_evt)
                        templates[e_width] = temp_evt.get_station(sid).get_channel(cid).get_trace()
                if templates != {}:
                    save_dic_help[np.deg2rad(cra)] = templates
            if save_dic_help != {}:
                save_dic[np.deg2rad(crz)] = save_dic_help

        # write as pickle file
        with open(os.path.join(self.__template_save_path, template_filename), "wb") as pickle_file:
            pickle.dump([save_dic], pickle_file)
            self.logger.info(f"The templates are saved to {os.path.join(self.__template_save_path, template_filename)}")
        if return_templates:
            return template_events


def _gaussian_func(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def _create_Efield(detector:detector.generic_detector.GenericDetector, run_id:int, event_id:int, channel_id:int, station_id:int, 
                   station_time:datetime.datetime, trace_samples:int, gaussian_width:float, e_phi:float, e_theta:float, cr_zenith:float, 
                   cr_azimuth:float, sampling_rate:float, debug:bool) -> Event:
    """ function that creates an event with a gaussian electric field """
    event = Event(run_id, event_id)

    station = Station(station_id)
    event.set_station(station)

    station.set_station_time(astropy.time.Time(station_time))

    sim_station = SimStation(station_id)
    station.set_sim_station(sim_station)

    sim_channel = SimChannel(channel_id, 0, 0)
    sim_station.add_channel(sim_channel)

    detector.set_event(run_id, event_id)

    electric_field = ElectricField([channel_id])

    e_field = [np.zeros(trace_samples), np.zeros(trace_samples), np.zeros(trace_samples)]
    x_data = np.arange(0, trace_samples, 1)
    for ii, x in enumerate(x_data):
        e_field[1][ii] = _gaussian_func(x, e_theta, 1000, gaussian_width)
        e_field[2][ii] = _gaussian_func(x, e_phi, 1000, gaussian_width)

    e_field = np.asarray(e_field)
    electric_field.set_trace(e_field, sampling_rate=sampling_rate)
    sim_station.add_electric_field(electric_field)

    if debug:
        # plot the electric field
        plt.plot(electric_field.get_times() / units.ns, electric_field.get_trace()[0])
        plt.plot(electric_field.get_times() / units.ns, electric_field.get_trace()[1])
        plt.plot(electric_field.get_times() / units.ns, electric_field.get_trace()[2])
        plt.xlabel('time [ns]')
        plt.ylabel('electric field')
        plt.show()

    sim_station.set_is_cosmic_ray()

    val_zenith = cr_zenith * (np.pi / 180)
    val_azimuth = cr_azimuth * (np.pi / 180)

    sim_station.set_parameter(stationParameters.zenith, val_zenith)
    sim_station.set_parameter(stationParameters.azimuth, val_azimuth)
    electric_field.set_parameter(electricFieldParameters.ray_path_type, 'direct')
    electric_field.set_parameter(electricFieldParameters.zenith, val_zenith)
    electric_field.set_parameter(electricFieldParameters.azimuth, val_azimuth)

    return event
