import NuRadioReco.detector.detector_base
import NuRadioReco.detector.generic_detector
import NuRadioReco.detector.RNO_G.rnog_detector
import json
import os


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
        Returns an Detector object.
        """
        
        if "source" in kwargs:
            source = kwargs["source"]
        else:
            source = "json"

        if source == "sql":
            return NuRadioReco.detector.detector_base.DetectorBase(
                json_filename=None, **kwargs)
            
        elif source == "mongo":
            kwargs.pop("source")
            return NuRadioReco.detector.RNO_G.rnog_detector.Detector(*args, **kwargs)
            
        elif source == "dictionary":
            
            if "dictionary" not in kwargs:
                raise ValueError("Argument \"dictionary\" is not passed to Detector().")
            
            station_dict = kwargs["dictionary"]
            filename = ''
            
        elif source == 'json':
            
            if len(args):
                json_filename = args[0]  # used to be passed as positional argument
            elif "json_filename" in kwargs:
                json_filename = kwargs.pop("json_filename")
            else:
                raise ValueError("Argument \"json_filename\" was not passed.")
            
            filename = find_path(json_filename)
            
            f = open(filename, 'r')
            station_dict = json.load(f)
                
        else:
            raise ValueError('Source must be either json or dictionary!')

        has_reference_entry = find_reference_entry(station_dict)
   
        if source == 'json':
            f.close()

        if has_reference_entry:
            return NuRadioReco.detector.generic_detector.GenericDetector(
                json_filename=filename, **kwargs)
        else:
            return NuRadioReco.detector.detector_base.DetectorBase(
                json_filename=filename, **kwargs)
