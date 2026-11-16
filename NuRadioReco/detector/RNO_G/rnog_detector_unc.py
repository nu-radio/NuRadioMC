import glob
import os
import re
import datetime
import copy
import lzma
import json
import numpy as np
import NuRadioReco.detector.RNO_G.rnog_detector as rnog_detector

def _avmnpy(alpha, var, nom, y):
    """
    Add the scaled difference between a varied and a nominal quantity to ``y``, i.e. calculate
    ``y + alpha * (var - nom)``.

    Dictionaries and lists are recursed into (see `_avmnpy_dict` and `_avmnpy_list`), any other
    type is treated as a single quantity.

    Parameters
    ----------
    alpha: float
        Scale factor, i.e. the nuisance parameter value

    var: dict, list, float or np.ndarray
        Quantity in the varied detector description

    nom: dict, list, float or np.ndarray
        Same quantity in the nominal detector description

    y: dict, list, float or np.ndarray
        Quantity to which the scaled difference is added

    Returns
    -------
    y: dict, list, float or np.ndarray
        Updated quantity

    Notes
    -----
    Quantities which are neither dicts nor lists have to support addition, subtraction and
    multiplication (e.g. floats or numpy arrays), otherwise a `RuntimeError` is raised. The
    interpolation rule is selected based on the type of ``nom``.
    """

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

def _avmnpy_dict(alpha, var, nom, y):
    """
    Calculate ``y + alpha * (var - nom)`` for dictionaries by recursing into each entry.

    Parameters
    ----------
    alpha: float
        Scale factor, i.e. the nuisance parameter value

    var: dict
        Quantity in the varied detector description

    nom: dict
        Same quantity in the nominal detector description

    y: dict
        Quantity to which the scaled difference is added

    Returns
    -------
    y: dict
        Updated quantity

    Notes
    -----
    All three dictionaries are required to have identical keys.
    """
    if not (set(var.keys()) == set(nom.keys()) == set(y.keys())):
        raise RuntimeError("Error: require keys to be identical")
    for key in var.keys():
        y[key] = _avmnpy(alpha, var[key], nom[key], y[key])
    return y

def _avmnpy_list(alpha, var, nom, y):
    """
    Calculate ``y + alpha * (var - nom)`` for lists by recursing into each element.

    Parameters
    ----------
    alpha: float
        Scale factor, i.e. the nuisance parameter value

    var: list
        Quantity in the varied detector description

    nom: list
        Same quantity in the nominal detector description

    y: list
        Quantity to which the scaled difference is added

    Returns
    -------
    y: list
        Updated quantity

    Notes
    -----
    All three lists are required to have identical lengths.
    """
    if not (len(var) == len(nom) == len(y)):
        raise RuntimeError("Error: require lengths to be identical")
    for ind in range(len(y)):
        y[ind] = _avmnpy(alpha, var[ind], nom[ind], y[ind])
    return y

class NuisanceInterpolator():

    def __init__(self, nominal, variations: dict):
        """
        Linear interpolation of a detector description along a set of eigenvariations.

        Parameters
        ----------
        nominal: dict
            Nominal detector description, e.g. as returned by `Detector.get_station`

        variations: dict
            Eigenvariations, keyed by variation id. Each entry is a dict with the keys "up"
            and "dn", holding the detector description at +1 sigma and -1 sigma of the
            respective variation.
        """
        self.nominal = nominal
        self.variations = variations
        self.var_ids = list(sorted(list(set(self.variations.keys()))))
        self.num_vars = len(self.var_ids)
    
    def evaluate_at_nuisance(self, nuis):
        """
        Interpolate the detector description for a set of nuisance parameters.

        Parameters
        ----------
        nuis: array of floats
            Nuisance parameter values, one per variation, in units of sigma. They are ordered by
            ascending variation id, see ``self.var_ids``.

        Returns
        -------
        interpolated: dict
            Interpolated detector description

        Notes
        -----
        The interpolation is linear and independent for each entry, i.e. the result is
        ``nominal + sum(nuis * (variation - nominal))``. The "+1 sigma" variation is used for
        positive nuisance values and the "-1 sigma" variation for negative ones.
        """
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
        """
        Detector that can be varied continuously along a set of calibration eigenvariations.

        Parameters
        ----------
        detector_folder: str
            Folder containing the nominal and the varied detector descriptions

        prefix: str
            Common file name prefix of the detector descriptions. The files are expected to be
            named ``<prefix>_nom.json.xz`` and ``<prefix>_var_<i>__<sigma>_<up|dn>.json.xz``.
            (Default: "calib_nuradio")
        """
        self.detector_folder = detector_folder
        self.prefix = prefix

        self.detector_time = None
        self.interpolator = None
        self.interpolated_detector = None

    def update(self, detector_time):
        """
        Load all detector descriptions and set up the interpolation.

        Parameters
        ----------
        detector_time: `datetime.datetime`
            Detector time at which the detector descriptions are evaluated

        Notes
        -----
        After this call all nuisance parameters are zero, i.e. the detector corresponds to the
        nominal description. The names of the nuisance parameters are available via
        `get_nuisance_names`.
        """
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

            if var_type in var_objs[var_num]:
                raise RuntimeError(f"Error: duplicate file for variation {var_num} and type {var_type}")

            var_objs[var_num][var_type] = _get_interpolation_obj(path)

        self.interpolator = NuisanceInterpolator(nom_obj, var_objs)
        self.nuis_names = [f"var_{var_id}" for var_id in self.interpolator.var_ids]
        self.set_nuisance_values(np.zeros(self.get_num_nuisances()))

    def set_nuisance_values(self, nuis):
        """
        Interpolate the detector description for a set of nuisance parameters.

        Parameters
        ----------
        nuis: array of floats
            Nuisance parameter values, one per variation, in units of sigma. They are ordered by
            ascending variation id, i.e. aligned with `get_nuisance_names`.
        """
        self.interpolated_detector = rnog_detector.Detector.from_dict(self.interpolator.evaluate_at_nuisance(nuis))
        self.interpolated_detector.update(self.detector_time)

    def as_detector(self):
        """
        Returns
        -------
        detector: `NuRadioReco.detector.RNO_G.rnog_detector.Detector`
            Detector interpolated for the current nuisance parameter values
        """
        return self.interpolated_detector

    def get_nuisance_names(self):
        """
        Returns
        -------
        nuisance_names: list of str
            Names of the nuisance parameters, sorted by variation id. This is the order expected
            for the ``nuis`` argument of `set_nuisance_values`.
        """
        return self.nuis_names

    def get_num_nuisances(self):
        """
        Returns
        -------
        num_nuisances: int
            Number of nuisance parameters
        """
        return len(self.get_nuisance_names())

    # forward these calls to rnog_detector.Detector

    def get_station(self, station_id):
        """ Return the station description, forwarded to `Detector.get_station`. """
        return self.as_detector().get_station(station_id)

    def get_channel(self, station_id, channel_id):
        """ Return the channel description, forwarded to `Detector.get_channel`. """
        return self.as_detector().get_channel(station_id, channel_id)

    def get_time_delay(self, station_id, channel_id):
        """ Return the time delay of a channel, forwarded to `Detector.get_time_delay`. """
        return self.as_detector().get_time_delay(station_id, channel_id)

    def get_relative_position(self, station_id, channel_id):
        """ Return the position of a channel, forwarded to `Detector.get_relative_position`. """
        return self.as_detector().get_relative_position(station_id, channel_id)

    def get_ice_model(self, station_id):
        """ Return the ice model of a station, forwarded to `Detector.get_ice_model`. """
        return self.as_detector().get_ice_model(station_id)

if __name__ == "__main__":

    import sys

    path = sys.argv[1]
    detector_time = datetime.datetime(2023, 9, 1)
    station_id = 11
    channel_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 22, 23]

    import itertools
    def _test_identical(det_a, det_b, getter, station_ids, channel_ids):
        for station_id, channel_id in itertools.product(station_ids, channel_ids):
            val_a = getattr(det_a, getter)(station_id, channel_id)
            val_b = getattr(det_b, getter)(station_id, channel_id)
            if not np.allclose(val_a, val_b):
                raise RuntimeError(f"Error: {val_a} vs {val_b} in "\
                        "{getter}(station_id = {station_id}, channel_id = {channel_id}")

    def _test_identical_if_interpolated(det, det_nom, det_var, alpha, getter, station_ids, channel_ids):
        for station_id, channel_id in itertools.product(station_ids, channel_ids):
            val = getattr(det, getter)(station_id, channel_id)
            val_nom = getattr(det_nom, getter)(station_id, channel_id)
            val_var = getattr(det_var, getter)(station_id, channel_id)
            val_int = val_nom + (val_var - val_nom) * alpha
            if not np.allclose(val, val_int):
                raise RuntimeError(f"Error: {val} vs {val_int} in "\
                        "{getter}(station_id = {station_id}, channel_id = {channel_id}")

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

    # Check if they are indeed identical
    _test_identical(det, nom_det, "get_time_delay", [station_id], channel_ids)
    _test_identical(det, nom_det, "get_relative_position", [station_id], channel_ids)

    # Move the first nuisance parameter to +1, which is identical to the "var0up" detector
    nuis[0] = 1.0
    det.set_nuisance_values(nuis)
    _test_identical(det, var0up_det, "get_time_delay", [station_id], channel_ids)
    _test_identical(det, var0up_det, "get_relative_position", [station_id], channel_ids)

    # Move the first nuisance parameter to -1, which is identical to the "var0dn" detector
    nuis[0] = -1.0
    det.set_nuisance_values(nuis)
    _test_identical(det, var0dn_det, "get_time_delay", [station_id], channel_ids)
    _test_identical(det, var0dn_det, "get_relative_position", [station_id], channel_ids)

    # Move the first nuisance paramter to 0.5, which results in a linear interpolation between the 'nominal'
    # and the 'var0up' detector.
    nuis[0] = 0.5
    det.set_nuisance_values(nuis)
    _test_identical_if_interpolated(det, nom_det, var0up_det, 0.5, "get_time_delay", [station_id], channel_ids)
    _test_identical_if_interpolated(det, nom_det, var0up_det, 0.5, "get_relative_position", [station_id], channel_ids)

    # Move the first nuisance paramter to -0.5, which results in a linear interpolation between the 'nominal'
    # and the 'var0dn' detector.
    nuis[0] = -0.5
    det.set_nuisance_values(nuis)
    _test_identical_if_interpolated(det, nom_det, var0dn_det, 0.5, "get_time_delay", [station_id], channel_ids)
    _test_identical_if_interpolated(det, nom_det, var0dn_det, 0.5, "get_relative_position", [station_id], channel_ids)
