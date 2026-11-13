import glob, os, re, datetime
import numpy as np
import NuRadioReco.detector.RNO_G.rnog_detector as rnog_detector

class NuisanceInterpolator():

    def __init__(self, nominal, variations: dict):
        self.nominal = nominal
        self.variations = variations
    
    def evaluate_at_nuisance(self, nuis):
        # TODO: interpolation logic here
        return self.nominal

class UncertaintyAwareDetector():

    def __init__(self, detector_folder, prefix = "calib_nuradio"):
        self.detector_folder = detector_folder
        self.prefix = prefix

        self.nuis_names = []

        self.interpolator = None
        self.interpolated_detector = None

    def update(self, detector_time):

        def _get_interpolation_obj(path):
            det = rnog_detector.Detector(detector_file = path)
            det.update(detector_time)
            obj = {station_id: det.get_station(station_id) for station_id in det.get_station_ids()}
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
            print(path)
            var_num, var_dir = _get_var_desc(path)

            if var_num not in var_objs:
                var_objs[var_num] = {}
                self.nuis_names.append(f"var_{var_num}")

            if var_dir in var_objs[var_num]:
                raise RuntimeError(f"Error: duplicate file for variation {var_num} and direction {var_dir}")

            var_objs[var_num][var_dir] = _get_interpolation_obj(path)

        self.interpolator = NuisanceInterpolator(nom_obj, var_objs)
        self.set_nuisance_values(np.zeros(self.get_num_nuisances()))

    def set_nuisance_values(self, nuis):
        self.interpolated_detector = rnog_detector.from_dict(self.interpolator.evaluate_at_nuisance(nuis))

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

    def get_relative_position(self, station_id, channel_id):
        return self.as_detector().get_relative_position(station_id, channel_id)

if __name__ == "__main__":
    path = "/cephfs/users/pwindi/tmp/nuradio_unc_test"
    det = UncertaintyAwareDetector(path)
    det.update(datetime.datetime(2023, 9, 1))
    stat = det.get_station(11)

    print(stat.keys())
