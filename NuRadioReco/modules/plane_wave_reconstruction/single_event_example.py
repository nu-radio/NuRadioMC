import logging
import os
import warnings
from datetime import datetime
from reconstruction import (
    PlaneWaveReconstructor,
    plot_skymap,
)
from travel_time_maps import map_from_dict

from NuRadioMC.utilities import medium
from NuRadioReco.detector.RNO_G.rnog_detector import Detector
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.modules.channelSignalReconstructor import (
    channelSignalReconstructor,
)
from NuRadioReco.modules.RNO_G.dataProviderRNOG import dataProviderRNOG
from NuRadioReco.utilities.logging import set_general_log_level

logging.getLogger("NuRadioMC").setLevel(logging.ERROR)
logging.getLogger("NuRadioMC.analytic_ray_tracing").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


set_general_log_level(logging.CRITICAL)
warnings.filterwarnings("ignore")




station_ = 11 
run_id = 4710
selected_events = range(3475, 3486)
skymap_outdir = './skymaps'
os.makedirs(skymap_outdir, exist_ok=True)

RNOG_DATA = "/pnfs/ifh.de/acs/radio/diskonly/data/full/root/"
run_path = os.path.join(RNOG_DATA, f"station{station_}/run{run_id}")

detector = Detector(select_stations=station_)

detector.update(datetime(2024, 1, 2, 0, 0, 0))
ice_model = medium.greenland_3exp_layered()
# alternative, use calibrated ice model and detector positions
#det_file = "/cvmfs/rnog.opensciencegrid.org/calibration/latest/station_11.json.xz"
#detector = Detector(detector_file=det_file, select_stations=station_)
#ice_model = _load_ice_model(det_file)

channels = [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23]

# set up all the standard NuRadio modules
provider = dataProviderRNOG()
plane_wave_reconstructor = PlaneWaveReconstructor(
    station=station_,
    detector=detector,
    channels=channels,
    ice_model=ice_model,
)
plane_wave_reconstructor.begin()

prep_map = plane_wave_reconstructor.tt_map

reader = provider.reader
reader.logger.setLevel(60)
logging.getLogger("NuRadioMC").setLevel(logging.ERROR)
logging.getLogger("NuRadioReco").setLevel(logging.CRITICAL)

signal_reconstructor = channelSignalReconstructor()

# initialize all the NuRadio modules

reader_kwargs = {
    "mattak_kwargs": {"backend": "uproot"},
    "apply_baseline_correction": None,
}
reader.begin([run_path], **reader_kwargs)

provider.channelBlockOffsetFitter.begin()
provider.channelGlitchDetector.begin()
provider.channelCableDelayAdder.begin()
signal_reconstructor.begin()

# begin loop over events
for event in reader.run():

    event_id = event.get_id()
    if selected_events is not None and event_id not in selected_events:
        continue
    print(f"Now on event {event_id}")

    station = event.get_station()
    trigger_type = station.get_first_trigger().get_name()
    station_time = station.get_station_time()
    
    signal_reconstructor.run(event, station, detector)
    SNRs = [station.get_channel(ch)[chp.SNR]["peak_2_peak_amplitude"] for ch in channels]

    (zenith, azimuth, correlation), corr_map = plane_wave_reconstructor.run(
        event, station, return_corr_map=True
    )
    
    print(f"Event {event_id}: zenith={zenith}, azimuth={azimuth}, correlation={correlation:.2f}")
    
    skymap_path = os.path.join(skymap_outdir, 
                                    f"station_{station_}_run_{run_id}_evt_{event_id}_corr_{correlation:.2f}.pdf")
    plot_skymap(skymap_path, corr_map, prep_map["zeniths"], prep_map["azimuths"])

signal_reconstructor.end()
provider.end()
plane_wave_reconstructor.end()
