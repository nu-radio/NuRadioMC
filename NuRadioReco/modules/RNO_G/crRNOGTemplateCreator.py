import os
import sys
import timeit
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units
import astropy
import numpy as np
import NuRadioReco.modules.io.coreas.readCoREASShower
from NuRadioReco.detector.generic_detector import GenericDetector
import NuRadioReco.modules.efieldToVoltageConverter
import logging
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.io.NuRadioRecoio
import json
import scipy
from NuRadioReco.modules.base import module
import datetime
from NuRadioReco.modules.io import NuRadioRecoio
from NuRadioReco.framework.event import Event
from NuRadioReco.framework.station import Station
from NuRadioReco.framework.channel import Channel
from NuRadioReco.framework.sim_station import SimStation
from NuRadioReco.framework.sim_channel import SimChannel
from NuRadioReco.framework.parameters import stationParameters
from NuRadioReco.framework.parameters import electricFieldParameters
from NuRadioReco.framework.electric_field import ElectricField
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelBandPassFilter
import pickle
import h5py
from tqdm import tqdm
from scipy import interpolate


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
        self.__Efield_width = None
        self.__Efield_amplitudes = None
        self.__template_event_id = None

        self.__cr_zenith = None
        self.__cr_azimuth = None

        self.__debug = None

        self.__template_save_path = None

        self.__efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
        self.__hardwareResponseIncorporator = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()
        self.__channelResampler = NuRadioReco.modules.channelResampler.channelResampler()

    def begin(self, detector_file, template_save_path='/home/henrichs/software/cr_analysis/artificial_template_bank/', debug=False, logger_level=logging.NOTSET):
        """
                begin method

                Parameters
                ----------
                detector_file: string
                    path to the detector file used for the template set
                template_save_path: string
                    path to a folder where the templates are stored
                logger_level: string or logging variable
                    Set verbosity level for logger (default: logging.NOTSET)
                """

        self.__detector_file = detector_file

        self.logger.setLevel(logger_level)

        # set up the efield to voltage converter
        self.__efieldToVoltageConverter.begin(debug=False)

        self.__debug = debug

        self.__template_save_path = template_save_path

    def set_template_parameter(self, template_run_id=[0, 0, 0], template_event_id=[0, 1, 2], template_station_id=[101, 101, 101], template_channel_id=[0, 0, 0], Efield_width=[5, 4, 2],
                                antenna_rotation=[160, 160, 160], Efield_amplitudes=[-0.2, 0.8], cr_zenith=[55, 55, 55], cr_azimuth=[0, 0, 0],
                                sampling_rate=3.2 * units.GHz, number_of_samples=2048):
        """
        set_parameter_templates method

        sets the parameter to create the template set

        Parameters
        ----------
        template_run_id: list of int
            run ids of the artificial templates
        template_event_id: list of int
            event ids of the artificial templates
        template_station_id:
            station ids of the artificial templates
        template_channel_id:
            channel ids of the artificial templates
        Efield_width: list of int
            width (in samples) of the gaussian function used to create the Efield
        antenna_rotation: list of int
            rotation angle of the LPDA
        Efield_ratio: list
            array with the amplitudes of the Efield components [E_theta, E_phi]
        cr_zenith: list of int
            zenith angle of the cr for the template
        cr_azimuth: list of int
            azimuth angle of the cr for the template
        sampling_rate: float
            sampling rate used to build the template
        number_of_samples: int
            number of samples used for the trace
        """

        self.__template_run_id = template_run_id
        self.__template_event_id = template_event_id
        self.__template_station_id = template_station_id
        self.__template_channel_id = template_channel_id

        self.__Efield_width = Efield_width
        self.__Efield_amplitudes = Efield_amplitudes
        self.__antenna_rotation = antenna_rotation

        self.__sampling_rate = sampling_rate
        self.__template_sample_number = number_of_samples

        self.__cr_zenith = cr_zenith
        self.__cr_azimuth = cr_azimuth

    def run(self, template_filename='templates_cr_station_101.p', include_hardware_response=True, return_templates=False):
        """
        run method

        creates a pickle file with the Efield trace of the artificial templates

        """

        # if no parameters are set, the standard parameters are used
        if self.__Efield_width is None:
            self.set_template_parameter()

        template_events = []
        save_dic = {}
        for crz in list(set(self.__cr_zenith)):
            save_dic_help = {}
            for cra in list(set(self.__cr_azimuth)):
                # loop over the different antenna rotation angles:
                templates = {}
                for rid, eid, sid, cid, e_width, antrot, cr_zen, cr_az in zip(self.__template_run_id, self.__template_event_id, self.__template_station_id, self.__template_channel_id, self.__Efield_width, self.__antenna_rotation, self.__cr_zenith, self.__cr_azimuth):
                    if cr_zen == crz and cr_az == cra:
                        # create the detector
                        det_temp = GenericDetector(json_filename=self.__detector_file, antenna_by_depth=False, create_new=True)
                        det_temp.update(datetime.datetime(2025, 10, 1))
                        det_temp.get_channel(101, 0)['ant_rotation_phi'] = antrot

                        station_time = datetime.datetime(2025, 10, 1)

                        temp_evt = create_Efield(det_temp, rid, eid, cid, sid, station_time, self.__template_sample_number, e_width,
                                                 self.__Efield_amplitudes[1], self.__Efield_amplitudes[0], cr_zen, cr_az, self.__sampling_rate, self.__debug)

                        self.__efieldToVoltageConverter.run(temp_evt, temp_evt.get_station(sid), det_temp)

                        if include_hardware_response:
                            self.__hardwareResponseIncorporator.run(temp_evt, temp_evt.get_station(sid), det_temp, sim_to_data=True)

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
        pickle.dump([save_dic], open(self.__template_save_path + template_filename, "wb"))
        if return_templates:
            return template_events


def gaussian_func(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def create_Efield(detector, run_id, event_id, channel_id, station_id, station_time, trace_samples, gaussian_width, e_phi, e_theta, cr_zenith, cr_azimuth, sampling_rate, debug):
    event = Event(run_id, event_id)

    station = Station(station_id)
    event.set_station(station)

    station.set_station_time(astropy.time.Time(station_time))

    channel = Channel(channel_id)
    station.add_channel(channel)

    sim_station = SimStation(station_id)
    station.set_sim_station(sim_station)

    sim_channel = SimChannel(channel_id, 0, 0)
    sim_station.add_channel(sim_channel)

    detector.set_event(run_id, event_id)

    electric_field = ElectricField([channel_id])

    e_field = [np.zeros(trace_samples), np.zeros(trace_samples), np.zeros(trace_samples)]
    x_data = np.arange(0, trace_samples, 1)
    for ii, x in enumerate(x_data):
        e_field[1][ii] = gaussian_func(x, e_theta, 1000, gaussian_width)
        e_field[2][ii] = gaussian_func(x, e_phi, 1000, gaussian_width)

    e_field = np.asarray(e_field)
    electric_field.set_trace(e_field, sampling_rate=sampling_rate)
    sim_station.add_electric_field(electric_field)

    # add DEBUG plot of electric fields!!!!!!!!!!!!!!!!!!!!!!!
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
