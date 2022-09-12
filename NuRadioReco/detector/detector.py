import NuRadioReco.detector.detector_base
import NuRadioReco.detector.generic_detector
import json
import os

class Detector(object):
    def __new__(
            cls,
            json_filename,
            source='json',
            dictionary=None,
            assume_inf=True,
            antenna_by_depth=True
    ):
        """
        Initialize the stations detector properties.
        This method will check if the JSON file containing the detector description is set up to be used by
        the DetectorBase or GenericDetector class and cause the correct class to be created.

        Parameters
        ----------
        json_filename : str
            the path to the json detector description file (if first checks a path relative to this directory, then a
            path relative to the current working directory of the user)
            default value is 'ARIANNA/arianna_detector_db.json'
        source: str
            'json', 'dictionary' or 'sql
            default value is 'json'
            If 'json' is passed, the JSON dictionary at the location specified
            by json_filename will be used
            If 'dictionary' is passed, the dictionary specified by the parameter
            'dictionary' will be used
            if 'sql' is specified, the file 'detector_sql_auth.json' file needs to be present in this folder that
            specifies the sql server credentials (see 'detector_sql_auth.json.sample' for an example of the syntax)
        dictionary: dict
            If 'dictionary' is passed to the parameter source, the dictionary
            passed to this parameter will be used for the detector description.
        assume_inf : Bool
            Default to True, if true forces antenna madels to have infinite boundary conditions, otherwise the antenna
            madel will be determined by the station geometry.
        antenna_by_depth: bool (default True)
            if True the antenna model is determined automatically depending on the depth of the antenna.
            This is done by appending e.g. '_InfFirn' to the antenna model name.
            if False, the antenna model as specified in the database is used.
        """
        if source == 'json':
            dir_path = os.path.dirname(os.path.realpath(__file__))  # get the directory of this file
            filename = os.path.join(dir_path, json_filename)
            if not os.path.exists(filename):
                # try local folder instead
                filename2 = json_filename
                if not os.path.exists(filename2):
                    raise NameError("can't locate json database file {} or {}".format(filename, filename2))
                filename = filename2

            f = open(filename, 'r')
            station_dict = json.load(f)
        elif source == 'dictionary':
            station_dict = dictionary
            filename = ''
        elif source == 'sql':   # Only the DetectorBaseClass can handle SQL, so no need to check for reference stations.
            det = object.__new__(NuRadioReco.detector.detector_base.DetectorBase)
            det.__init__(
                json_filename=None,
                source=source,
                dictionary=dictionary,
                assume_inf=assume_inf,
                antenna_by_depth=antenna_by_depth
            )
            return det
        else:
            raise ValueError('Source must be either json or dictionary!')

        reference_entry_found = False
        for station in station_dict['stations']:
            if 'reference_station' in station_dict['stations'][station].keys():
                reference_entry_found = True
                break
        for channel in station_dict['channels']:
            if 'reference_channel' in station_dict['channels'][channel].keys() or 'reference_station' in \
                    station_dict['channels'][channel].keys():
                reference_entry_found = True
                break
        if source == 'json':
            f.close()

        if reference_entry_found:
            det = object.__new__(NuRadioReco.detector.generic_detector.GenericDetector)
            det.__init__(
                json_filename=filename,
                source=source,
                dictionary=dictionary,
                assume_inf=assume_inf,
                antenna_by_depth=antenna_by_depth
            )
        else:
            det =  object.__new__(NuRadioReco.detector.detector_base.DetectorBase)
            det.__init__(
                source=source,
                json_filename=filename,
                dictionary=dictionary,
                assume_inf=assume_inf,
                antenna_by_depth=antenna_by_depth
            )
        return det