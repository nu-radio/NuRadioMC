import glob
import os
import re
import datetime
import copy
import lzma
import json
import numpy as np
import NuRadioReco.detector.RNO_G.rnog_detector as rnog_detector

# calculates y = alpha * (var - nom) + y and returns y
def _avmnpy(alpha, var, nom, y):

    # Bail out early if nothing needs to be done
    if alpha == 0.0:
        return y
    if var == nom:
        return y

    if isinstance(nom, dict):
        if not (isinstance(var, dict) and isinstance(y, dict)):
            raise RuntimeError("Error: cannot interpolate---incompatible structures.")
        y = _avmnpy_dict(alpha, var, nom, y)
    elif isinstance(nom, list):
        if not (isinstance(var, list) and isinstance(y, list)):
            raise RuntimeError("Error: cannot interpolate---incompatible structures.")
        y = _avmnpy_list(alpha, var, nom, y)
    else:
        # everything else
        try: 
            y += alpha * (var - nom)
        except:
            raise RuntimeError(f"Error: dedicated interpolation rule needed for '{type(y)}'---please implement!")

    return y

# calculates y = alpha * (var - nom) + y for dictionaries
def _avmnpy_dict(alpha, var, nom, y):
    if not (set(var.keys()) == set(nom.keys()) == set(y.keys())):
        raise RuntimeError("Error: require keys to be identical")
    for key in var.keys():
        y[key] = _avmnpy(alpha, var[key], nom[key], y[key])
    return y

# calculates y = alpha * (var - nom) + y for lists
def _avmnpy_list(alpha, var, nom, y):
    if not (len(var) == len(nom) == len(y)):
        raise RuntimeError("Error: require lengths to be identical")
    for ind in range(len(y)):
        y[ind] = _avmnpy(alpha, var[ind], nom[ind], y[ind])
    return y

class NuisanceInterpolator():

    def __init__(self, nominal, variations: dict):
        self.nominal = nominal
        self.variations = variations
        self.var_ids = list(set(self.variations.keys()))
        self.num_vars = len(self.var_ids)
    
    def evaluate_at_nuisance(self, nuis):
        if len(nuis) != self.num_vars:
            raise RuntimeError(f"Error: expect exactly {self.num_vars} nuisance parameters.")

        interpolated = copy.deepcopy(self.nominal)
        for cur_var_id, cur_nuis in zip(self.var_ids, nuis):
            if cur_nuis == 0.0:
                continue
            elif cur_nuis > 0.0:
                cur_var = self.variations[cur_var_id]["up"]
            else:
                cur_var = self.variations[cur_var_id]["dn"]

            interpolated = _avmnpy(abs(cur_nuis), cur_var, self.nominal, interpolated)

        return interpolated

class UncertaintyAwareDetector():

    def __init__(self, detector_folder, prefix = "calib_nuradio"):
        self.detector_folder = detector_folder
        self.prefix = prefix

        self.nuis_names = []

        self.detector_time = None
        self.interpolator = None
        self.interpolated_detector = None

    def update(self, detector_time):
        self.detector_time = detector_time

        def _get_interpolation_obj(path):
            with lzma.open(path, 'rt') as det_file:
                det_data = json.load(det_file)

            det = rnog_detector.Detector(detector_file = path)
            det.update(detector_time)
            data = {station_id: copy.deepcopy(det.get_station(station_id)) 
                    for station_id in det.get_station_ids()}
            obj = {
                "data": data,
                "version": copy.deepcopy(det_data["version"]),
                "periods": copy.deepcopy(det_data["periods"]),
                "default_values": copy.deepcopy(det_data["default_values"]),
                "additional_data": copy.deepcopy(det_data["additional_data"]),
                "comment": copy.deepcopy(det_data["comment"])
            }
            return obj

        def _get_var_desc(path):
            var_re = re.compile(f"{self.prefix}_var_(.+)__1_(.+)\\.json\\.xz")
            match = var_re.match(os.path.basename(path))
            if match is None:
                raise ValueError(f"Could not parse variation description from {path}")
            return int(match.group(1)), match.group(2)

        nom_path = os.path.join(self.detector_folder, self.prefix + "_nom.json.xz")
        nom_obj = _get_interpolation_obj(nom_path)

        var_objs = {}
        for path in glob.glob(os.path.join(self.detector_folder, self.prefix + "_var_*.json.xz")):
            var_num, var_type = _get_var_desc(path)

            if var_num not in var_objs:
                var_objs[var_num] = {}
                self.nuis_names.append(f"var_{var_num}")

            if var_type in var_objs[var_num]:
                raise RuntimeError(f"Error: duplicate file for variation {var_num} and type {var_type}")

            var_objs[var_num][var_type] = _get_interpolation_obj(path)

        self.interpolator = NuisanceInterpolator(nom_obj, var_objs)
        self.set_nuisance_values(np.zeros(self.get_num_nuisances()))
        self.nuis_names = sorted(self.nuis_names)

    def set_nuisance_values(self, nuis):
        self.interpolated_detector = rnog_detector.Detector.from_dict(self.interpolator.evaluate_at_nuisance(nuis))
        self.interpolated_detector.update(self.detector_time)

    def as_detector(self):
        return self.interpolated_detector

    def get_nuisance_names(self):
        return self.nuis_names

    def get_num_nuisances(self):
        return len(self.get_nuisance_names())

    # forward these calls to rnog_detector.Detector

    def get_station(self, station_id):
        return self.as_detector().get_station(station_id)

    def get_channel(self, station_id, channel_id):
        return self.as_detector().get_channel(station_id, channel_id)

    def get_time_delay(self, station_id, channel_id):
        return self.as_detector().get_time_delay(station_id, channel_id)

    def get_relative_position(self, station_id, channel_id):
        return self.as_detector().get_relative_position(station_id, channel_id)

    def get_ice_model(self, station_id):
        return self.as_detector().get_ice_model(station_id)

if __name__ == "__main__":

    path = "/cephfs/users/pwindi/tmp/nuradio_unc_test"
    detector_time = datetime.datetime(2023, 9, 1)
    station_id = 11
    channel_id = 5

    # Load the detector that is able to interpolate
    det = UncertaintyAwareDetector(path)
    det.update(detector_time)

    # For comparisons, separately load the nominal detector and the +/- 1-sigma variations for variation 0
    nom_det = rnog_detector.Detector(detector_file = os.path.join(path, "calib_nuradio_nom.json.xz"))
    nom_det.update(detector_time)
    var0up_det = rnog_detector.Detector(detector_file = os.path.join(path, "calib_nuradio_var_0__1_up.json.xz"))
    var0up_det.update(detector_time)
    var0dn_det = rnog_detector.Detector(detector_file = os.path.join(path, "calib_nuradio_var_0__1_dn.json.xz"))
    var0dn_det.update(detector_time)

    # Set all nuisance parameters to zero, which corresponds to the nominal detector
    nuis = np.zeros(det.get_num_nuisances())
    det.set_nuisance_values(nuis)
    print(det.get_ice_model(station_id).get_index_of_refraction([0, 0, -30]))
    print(nom_det.get_ice_model(station_id).get_index_of_refraction([0, 0, -30]))
    print(det.get_time_delay(station_id, channel_id))
    print(nom_det.get_time_delay(station_id, channel_id))

    nuis[0] = 1.0
    det.set_nuisance_values(nuis)
    print(det.get_ice_model(station_id).get_index_of_refraction([0, 0, -30]))
    print(var0up_det.get_ice_model(station_id).get_index_of_refraction([0, 0, -30]))
    print(det.get_time_delay(station_id, channel_id))
    print(var0up_det.get_time_delay(station_id, channel_id))

    nuis[0] = 0.5
    det.set_nuisance_values(nuis)
    print(det.get_ice_model(station_id).get_index_of_refraction([0, 0, -30]))
    print(var0up_det.get_ice_model(station_id).get_index_of_refraction([0, 0, -30]))
    print(det.get_time_delay(station_id, channel_id))
    print(var0up_det.get_time_delay(station_id, channel_id))
