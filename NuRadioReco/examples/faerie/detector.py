from NuRadioReco.utilities import units
import numpy as np


class FAERIEDetector():

    def __init__(self):
        pass

    def set_event(self, evt):
        self.event = evt

    def get_station_ids(self):
        """ Returns all station ids """
        return self.event.get_station_ids()

    def get_channel_ids(self, station_id):
        """ Returns all channel ids of one station (sorted) """
        station = self.event.get_station(station_id)
        return np.unique([efield.get_channel_ids() for efield in station.get_sim_station().get_electric_fields()])

    def get_relative_position(self, station_id, channel_id):
        """ Return the relative position of the antenna in the station (relative to station position) """
        sim_station = self.event.get_station(station_id).get_sim_station()

        efield_position = np.unique([efield.get_position() for efield in sim_station.get_electric_fields_for_channels([channel_id])], axis=0)
        assert len(efield_position) == 1, "There should be only one unique position for each channel"

        return efield_position[0]

    ### Constant Returns ###

    def get_absolute_position(self, station_id):
        """ Return the station position """
        return np.array([0, 0, 0])

    # def get_cable_delay(self, station_id=None, channel_id=None):
    #     return 0

    # def get_site(self, station_id=None):
    #     return "summit"

    # def get_antenna_model(self, station_id=None, channel_id=None, zenith_antenna=None):
    #     """ Returns the antenna model """
    #     return "RNOG_vpol_4inch_center_n1.73"

    # def get_antenna_orientation(self, station_id=None, channel_id=None):
    #     """ Returns the channel's 4 orientation angles in rad """
    #     return np.deg2rad([0, 0, 90, 90])

    # def get_site_coordinates(self, station_id=None):
    #     """ Returns latitude and longitude of SKA in degrees """
    #     return 72.57, -38.46

    # def get_number_of_samples(self, station_id=None, channel_id=None):
    #     return 2048

    # def get_sampling_frequency(self, station_id=None, channel_id=None):
    #     return 3.2 * units.GHz
