import json
import os
import numpy as np
import logging

from NuRadioReco.detector import detector_base
from NuRadioReco.detector import generic_detector
from NuRadioReco.detector.RNO_G import rnog_detector



def find_path(name):
    """ Checks for file relative to local folder and detector.py """
    dir_path = os.path.dirname(os.path.realpath(__file__))  # get the directory of this file
    filename = os.path.join(dir_path, name)
    if os.path.exists(filename):
        return filename
    else:
        # try local folder instead
        if not os.path.exists(name):
            raise NameError("Can't locate json database file in: {} or {}".format(filename, name))
        return name


def find_reference_entry(station_dict):
    """
    Search for the strings "reference_station" or "reference_channel" in the detector description.
    This is used to determine whether to use the detector_base or generic_detector class.
    """

    for station in station_dict['stations']:
        if 'reference_station' in station_dict['stations'][station].keys():
            return True

    for channel in station_dict['channels']:
        if 'reference_channel' in station_dict['channels'][channel].keys() or 'reference_station' in \
                station_dict['channels'][channel].keys():
            return True

    return False


def Detector(*args, **kwargs):
        """
        This function returns a detector class object. It chooses the correct class based on the "source" argument.
        The returned object is of one of these classes:

            - kwargs["source'] == "rnog_mongo" -> `NuRadioReco.detector.RNO_G.rnog_detector`
            - kwargs["source'] == "sql" -> `NuRadioReco.detector.detector_base`
            - kwargs["source'] == "json" or "dictionary" -> `NuRadioReco.detector.detector_base` or
                                                            `NuRadioReco.detector.generic_detector`

        For 'kwargs["source'] == "json"', whether to use "detector_base" or "generic_detector"
        depends on whether a reference station / channel is defined in the json file / dictionary
        or not.

        Parameters
        ----------

        args: Positional arguments (arguments without keyword)
            For backwards compatibility, when source is sql | json | dictionary, args are interpreted as follows:

                - json_filename = args[0] only when source == "json")
                - source = args[1]
                - dictionary = args[2]
                - assume_inf = args[3]
                - antenna_by_depth = args[4]

        kwargs: Optional arguments (arguments with keyword)
            Keyword arguments passed to detector object. The argument "source" is used to select the
            correct class (see description above). If no keyword "source" is passed, the default "json"
            is used.

        Returns
        -------

        det: NuRadioReco.detector.* (see options above)
            Detector class object
        """

        # Interprete positional arguments (args) for backwards compatibility
        # when source is sql | json | dictionary
        # json_filename = args[0] is used below (when source == 'json').
        if len(args) >= 2:
            source = args[1].lower()
        else:
            source = kwargs.pop("source", "json").lower()

        if len(args) >= 3:
            dictionary = args[2]
        else:
            dictionary = kwargs.pop("dictionary", None)

        if len(args) >= 4:
            assume_inf = args[3]
        else:
            assume_inf = kwargs.pop("assume_inf", True)

        if len(args) >= 5:
            antenna_by_depth = args[4]
        else:
            antenna_by_depth = kwargs.pop("antenna_by_depth", True)


        if source == "sql":
            return detector_base.DetectorBase(
                json_filename=None, source=source, dictionary=dictionary,
                assume_inf=assume_inf, antenna_by_depth=antenna_by_depth)

        elif source == "rnog_mongo":
            return rnog_detector.Detector(*args, **kwargs)

        elif source == "dictionary":

            if "dictionary" not in kwargs:
                raise ValueError("Argument \"dictionary\" is not passed to Detector() while source=\"dictionary\" is set.")

            station_dict = kwargs["dictionary"]
            filename = ''

        elif source == 'json':

            if len(args):
                json_filename = args[0]  # used to be passed as positional argument
            elif "json_filename" in kwargs:
                json_filename = kwargs.pop("json_filename")
            else:
                raise ValueError("No possitional arguments and no argument \"json_filename\" "
                                 "was not passed while source=\"json\" (default) is set.")

            filename = find_path(json_filename)

            f = open(filename, 'r')
            station_dict = json.load(f)

        else:
            raise ValueError(f'Unknown source specifed (\"{source}\"). '
                             f'Must be one of \"json\", \"sql\", "\dictionary\", \"mongo\"')

        has_reference_entry = find_reference_entry(station_dict)

        if source == 'json':
            f.close()

        has_default = np.any([arg in kwargs and kwargs[arg] is not None
                              for arg in ["default_station", "default_channel", "default_device"]])

        if has_reference_entry or has_default:
            if has_default:
                logging.warning(
                    'Deprecation warning: Passing the default detector station is deprecated. Default stations and default'
                    'channel should be specified in the detector description directly.')

                if "default_station" in kwargs:
                    logging.info('Default detector station provided (station '
                                 f'{kwargs["default_station"]}) -> Using generic detector')

            return generic_detector.GenericDetector(
                json_filename=filename, source=source, dictionary=dictionary,
                assume_inf=assume_inf, antenna_by_depth=antenna_by_depth, **kwargs)
        else:
            # Keys might be present (but should be None). Keys are deprecated, keep them for backwards compatibility
            for key in ["default_station", "default_channel", "default_device"]:
                kwargs.pop(key, "None")

            return detector_base.DetectorBase(
                json_filename=filename, source=source, dictionary=dictionary,
                assume_inf=assume_inf, antenna_by_depth=antenna_by_depth, **kwargs)
