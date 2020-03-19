import numpy as np
from NuRadioReco.utilities import units
from NuRadioReco.utilities.io_utilities import read_pickle
import logging
from six import iteritems
import os


class Templates(object):
    __instance = None

    def __new__(cls, path):
        if Templates.__instance is None:
            Templates.__instance = object.__new__(cls)
        return Templates.__instance

    def __init__(self, path):
        self.__ref_cr_templates = {}
        self.__cr_templates = {}
        self.__cr_template_set = {}
        self.__cr_template_set_full = {}
        self.__ref_nu_templates = {}
        self.__nu_templates = {}
        self.__nu_template_set = {}
        self.__path = path
        self.logger = logging.getLogger("NuRadioReco.Templates")

    def set_template_directory(self, path):
        self.__path = path

    def __load_cr_template(self, station_id):
        path = os.path.join(self.__path, 'templates_cr_station_{}.pickle'.format(station_id))
        if(os.path.exists(path)):
            self.__cr_templates[station_id] = read_pickle(path)
            zen_ref = np.deg2rad(60)
            az_ref = np.deg2rad(0)
            self.__ref_cr_templates[station_id] = self.__cr_templates[station_id][0][zen_ref][az_ref]
        else:
            self.logger.error("template file {} not found".format(path))
            raise IOError

    def get_cr_ref_templates(self, station_id):
        """
        returns one cosmic ray template per channel for the reference direction
        """
        if station_id not in self.__ref_cr_templates.keys():
            self.__load_cr_template(station_id)
        return self.__ref_cr_templates[station_id]

    def get_cr_ref_template(self, station_id):
        """
        returns one cosmic ray template for the reference direction, the same for all channels
        """
        self.logger.info("Getting template for station ID {}".format(station_id))
        if station_id in [51, 52]:
            tmpl = self.get_cr_ref_templates(station_id)[4]  # FIXME: hardcoded that channel 4 is a cosmic-ray sensitive channel
        elif station_id == 32:
            tmpl = self.get_cr_ref_templates(station_id)[1]
        elif station_id == 61:
            tmpl = self.get_cr_ref_templates(station_id)[5]
        else:
            self.logger.error("Provided value for CR senistive channel of station {} in templates.py".format(station_id))
            tmpl = None

        return tmpl

    def get_set_of_cr_templates_full(self, station_id, n=100):
        """
        gets set of n templates to allow for the calculation of average templates
        """
        if station_id not in self.__ref_cr_templates.keys():
            self.__load_cr_template(station_id)
        if self.__cr_template_set_full == {}:
            self.logger.info("Getting set of templates for station ID {}".format(station_id))
            cr_set = {}
            n_tmpl = 0

            for templates in self.__cr_templates[station_id]:
                for zen, zen_tempaltes in templates.items():
                    for az, template in zen_tempaltes.items():
                        cr_set[n_tmpl] = template
                        n_tmpl += 1
                        if n_tmpl >= n:
                            self.__cr_template_set_full = cr_set
                            return self.__cr_template_set_full
            self.logger.warning(f"{n} templates requested but only {n_tmpl} are available. Returning only {n_tmpl} templates.")
            return self.__cr_template_set_full
        else:
            return self.__cr_template_set_full

    def get_set_of_cr_templates(self, station_id, n=100):
        """
        gets set of n templates to allow for the calculation of average templates

        loops first over different coreas pulses with different frequency content
        and then over azimuth angles of 0, 22.5 and 45 degree
        and then over zenith angles of 60, 50 and 70 degree
        """
        if station_id not in self.__ref_cr_templates.keys():
            self.__load_cr_template(station_id)
        if self.__cr_template_set == {}:
            self.logger.info("Getting set of templates for station ID {}".format(station_id))
            n_tmpl = 0
            zen_refs = np.deg2rad([60, 50, 70])
            az_refs = np.deg2rad([0, 22.5, 45])

            for zen in zen_refs:
                for az in az_refs:
                    for templates in self.__cr_templates[station_id]:
                        self.__cr_template_set[n_tmpl] = templates[zen][az]
                        n_tmpl += 1
                        if n_tmpl >= n:
                            break
        return self.__cr_template_set

    def get_set_of_nu_templates(self, station_id, n=100):
        """
        gets set of n templates to allow for the calculation of average templates

        loops first over different viewing angles
        and then over azimuth angles of 0, 22.5 and 45 degree
        and then over zenith angles of 100, 120 and 140 degree
        """
        if station_id not in self.__ref_nu_templates.keys():
            self.__load_nu_template(station_id)
        if self.__nu_template_set == {}:
            self.logger.info("Getting set of templates for station ID {}".format(station_id))
            n_tmpl = 0
            zen_refs = np.deg2rad([100, 120, 140])
            az_refs = np.deg2rad([0, 22.5, 45])
            dCherenkovs = np.deg2rad([0, -0.5, -1, -1.5, -2, -3, -4, -5])

            for zen in zen_refs:
                for az in az_refs:
                    for dCh in dCherenkovs:
                        self.__nu_template_set[n_tmpl] = self.__nu_templates[station_id][zen][az][dCh]
                        n_tmpl += 1
                        if n_tmpl >= n:
                            break
        return self.__nu_template_set

    def __load_nu_template(self, station_id):
        path = os.path.join(self.__path, 'templates_nu_station_{}.pickle'.format(station_id))
        if(os.path.exists(path)):
            self.__nu_templates[station_id] = read_pickle(path)
            zen_ref = np.deg2rad(140)
            az_ref = np.deg2rad(45)
            self.__ref_nu_templates[station_id] = self.__nu_templates[station_id][zen_ref][az_ref]
        else:
            self.logger.error("template file {} not found".format(path))
            raise IOError

    def get_nu_ref_templates(self, station_id):
        """
        returns one neutrino template per channel for the reference direction and on the cherenkov cone
        """
        if station_id not in self.__ref_nu_templates.keys():
            self.__load_nu_template(station_id)
        return self.__ref_nu_templates[station_id][0.0]

    def get_nu_ref_template(self, station_id):
        """
        returns one neutrino template for the reference direction and on the cherenkov cone, the same for all channels
        """
        return self.get_nu_ref_templates(station_id)[0]  # FIXME: hardcoded that channel 0 is a neutrino sensitive channel
